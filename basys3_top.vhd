-- =============================================================================
-- basys3_top.vhd  --  MNIST ViT inference on Basys 3 (Artix-7 XC7A35T)
-- =============================================================================
-- Pipeline:
--   UART-RX/TX (115200 8N1)
--     -> pixel normalisation LUT (uint8 -> Q1.7 int8)
--     -> patch_embed  (28x28 pixels -> 16 x 32-dim tokens)
--     -> encoder_block(structural) + weight_mem  (1 transformer layer)
--     -> classifier  (GAP -> linear 32->10 -> argmax)
--     -> UART-TX (predicted class byte back to PC)
--     -> 7-segment display (predicted digit 0-9)
--
-- Basys 3 board connections:
--   clk      : W5   100 MHz crystal oscillator (input to MMCM only)
--   btnc     : U18  BTNC centre push-button (active-high reset)
--   uart_rxd : B18  USB-UART bridge RX (PC -> FPGA)
--   seg[6:0] : {CG,CF,CE,CD,CC,CB,CA} = {g,f,e,d,c,b,a}, active-low cathodes
--   an[3:0]  : digit anodes, active-low  (an[0] = rightmost)
--   led[3:0] : activity / status LEDs
--
-- Clock architecture:
--   100 MHz crystal -> MMCME2_BASE -> 20 MHz internal clock (clk20)
--   The softmax critical path is ~41 ns; 20 MHz (50 ns period) gives ~9 ns margin.
--   UART timing: CLKS_PER_BIT = 20_000_000 / 115_200 = 173 clocks/bit.
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;          -- elaboration-time only (constant gen)

library unisim;
use unisim.vcomponents.all;      -- MMCME2_BASE, BUFG

use work.clog2_pkg.all;

entity basys3_top is
    port (
        clk      : in  std_logic;                      -- 100 MHz  W5
        btnc     : in  std_logic;                      -- reset     U18

        uart_rxd : in  std_logic;                      -- UART RX   B18
        uart_txd : out std_logic;                      -- UART TX   A18

        seg      : out std_logic_vector(6 downto 0);   -- segments {g..a}
        dp       : out std_logic;                      -- decimal point
        an       : out std_logic_vector(3 downto 0);   -- anodes

        led      : out std_logic_vector(3 downto 0)    -- status LEDs
    );
end entity basys3_top;

architecture rtl of basys3_top is

    -- =========================================================================
    -- Model dimensions  (must match QAT export)
    -- =========================================================================
    constant DATA_WIDTH  : positive := 8;
    constant MODEL_DIM   : positive := 32;
    constant NUM_HEADS   : positive := 1;
    constant HEAD_DIM    : positive := 32;
    constant HIDDEN_DIM  : positive := 64;
    constant SEQ_LEN     : positive := 16;
    constant PATCH_SIZE  : positive := 7;
    constant IMG_SIZE    : positive := 28;
    constant N_CLS       : positive := 10;
    constant N_PX        : positive := IMG_SIZE * IMG_SIZE;   -- 784

    -- =========================================================================
    -- UART timing  (8N1, LSB-first)
    -- =========================================================================
    -- Clock is 20 MHz (50 ns period -- keeps softmax timing in budget).
    -- UART: CLKS_PER_BIT = 20_000_000 / 115_200 = 173 clocks/bit.
    constant CLK_HZ       : positive := 20_000_000;
    constant BAUD_RATE    : positive := 115_200;
    constant CLKS_PER_BIT : positive := CLK_HZ / BAUD_RATE;  -- 173

    -- =========================================================================
    -- Pixel normalisation LUT
    --   q = round( (p/255 - 0.1307) / 0.3081 * 128 ),  clamp [-128, 127]
    -- Built at elaboration time using ieee.math_real; zero runtime cost.
    -- =========================================================================
    type norm_lut_t is array (0 to 255) of std_logic_vector(DATA_WIDTH - 1 downto 0);

    function gen_norm_lut return norm_lut_t is
        variable lut : norm_lut_t;
        variable f   : real;
        variable q   : integer;
    begin
        for p in 0 to 255 loop
            f := (real(p) / 255.0 - 0.1307) / 0.3081 * 128.0;
            q := integer(round(f));
            if q >  127 then q :=  127; end if;
            if q < -128 then q := -128; end if;
            lut(p) := std_logic_vector(to_signed(q, DATA_WIDTH));
        end loop;
        return lut;
    end function;

    constant NORM_LUT : norm_lut_t := gen_norm_lut;

    -- =========================================================================
    -- Weight-memory address widths  (duplicated from synth_wrapper)
    -- =========================================================================
    constant AW_MHA : positive := clog2(MODEL_DIM * MODEL_DIM);    -- 10
    constant AW_W1  : positive := clog2(HIDDEN_DIM * MODEL_DIM);   -- 11
    constant AW_B1  : positive := clog2(HIDDEN_DIM);               --  6
    constant AW_W2  : positive := clog2(MODEL_DIM * HIDDEN_DIM);   -- 11
    constant AW_B2  : positive := clog2(MODEL_DIM);                --  5

    -- =========================================================================
    -- MMCM: 100 MHz -> 20 MHz
    --   VCO = 100 * 10 = 1000 MHz  (within Artix-7 range 600-1200 MHz)
    --   CLKOUT0 = 1000 / 50 = 20 MHz
    -- =========================================================================
    signal mmcm_fb      : std_logic;
    signal clk20_raw    : std_logic;
    signal clk20        : std_logic;   -- 20 MHz, BUFG-routed, used everywhere
    signal mmcm_locked  : std_logic;

    -- =========================================================================
    -- Reset synchroniser  (clocked on clk20, held until MMCM locked)
    -- =========================================================================
    signal rst_meta  : std_logic := '0';
    signal rst_sync  : std_logic := '0';
    signal rstn      : std_logic;

    -- =========================================================================
    -- UART-RX
    -- =========================================================================
    type uart_state_t is (U_IDLE, U_START, U_DATA, U_STOP);
    signal uart_state   : uart_state_t := U_IDLE;
    signal uart_clk_cnt : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal uart_bit_cnt : integer range 0 to 7 := 0;
    signal uart_shift   : std_logic_vector(7 downto 0) := (others => '0');
    signal uart_data    : std_logic_vector(7 downto 0) := (others => '0');
    signal uart_valid   : std_logic := '0';

    -- =========================================================================
    -- UART-TX  (one-byte response after inference)
    -- =========================================================================
    type uart_tx_state_t is (TX_IDLE, TX_START, TX_DATA, TX_STOP);
    signal uart_tx_state   : uart_tx_state_t := TX_IDLE;
    signal uart_tx_clk_cnt : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal uart_tx_bit_cnt : integer range 0 to 7 := 0;
    signal uart_tx_shift   : std_logic_vector(7 downto 0) := (others => '1');
    signal uart_tx_data    : std_logic_vector(7 downto 0) := (others => '0');
    signal uart_tx_req     : std_logic := '0';
    signal uart_tx_busy    : std_logic := '0';
    signal uart_txd_r      : std_logic := '1';

    -- =========================================================================
    -- Top-level control FSM
    -- =========================================================================
    type top_state_t is (
        TS_IDLE,         -- waiting for first UART byte
        TS_START_PULSE,  -- present pixel-0 to patch_embed (1 cycle after start)
        TS_COLLECT,      -- streaming pixels 1..783 to patch_embed
        TS_INFER,        -- waiting for classifier.done
        TS_SHOW          -- hold result until reset
    );
    signal top_state  : top_state_t := TS_IDLE;
    signal px_cnt     : integer range 0 to N_PX := 0;
    signal first_px   : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal result_cls : integer range 0 to N_CLS - 1 := 0;
    signal result_reg : integer range 0 to N_CLS - 1 := 0;
    signal result_sent : std_logic := '0';

    -- =========================================================================
    -- patch_embed I/O
    -- =========================================================================
    signal pe_start     : std_logic := '0';
    signal pe_done      : std_logic;
    signal pe_i_data    : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal pe_i_valid   : std_logic := '0';
    signal pe_o_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal pe_o_valid   : std_logic;
    signal pe_o_last    : std_logic;
    signal pe_o_channel : integer;

    -- =========================================================================
    -- encoder_block <-> weight_mem interconnect
    -- =========================================================================
    signal w_q_addr_s    : std_logic_vector(AW_MHA - 1 downto 0);
    signal w_q_data_s    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_q_re_s      : std_logic;

    signal w_k_addr_s    : std_logic_vector(AW_MHA - 1 downto 0);
    signal w_k_data_s    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_k_re_s      : std_logic;

    signal w_v_addr_s    : std_logic_vector(AW_MHA - 1 downto 0);
    signal w_v_data_s    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_v_re_s      : std_logic;

    signal w_o_addr_s    : std_logic_vector(AW_MHA - 1 downto 0);
    signal w_o_data_s    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_o_re_s      : std_logic;

    signal ffn_w1_addr_s : std_logic_vector(AW_W1 - 1 downto 0);
    signal ffn_w1_data_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_w1_re_s   : std_logic;

    signal ffn_b1_addr_s : std_logic_vector(AW_B1 - 1 downto 0);
    signal ffn_b1_data_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_b1_re_s   : std_logic;

    signal ffn_w2_addr_s : std_logic_vector(AW_W2 - 1 downto 0);
    signal ffn_w2_data_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_w2_re_s   : std_logic;

    signal ffn_b2_addr_s : std_logic_vector(AW_B2 - 1 downto 0);
    signal ffn_b2_data_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_b2_re_s   : std_logic;

    -- encoder_block output (-> classifier input)
    signal enc_o_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal enc_o_valid   : std_logic;
    signal enc_o_last    : std_logic;
    signal enc_o_channel : integer;

    -- Debug taps (tied off)
    signal enc_mha_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal enc_mha_valid   : std_logic;
    signal enc_mha_last    : std_logic;
    signal enc_mha_channel : integer;
    signal enc_ffn_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal enc_ffn_valid   : std_logic;
    signal enc_ffn_last    : std_logic;
    signal enc_ffn_channel : integer;

    -- =========================================================================
    -- classifier I/O
    -- =========================================================================
    signal cl_start  : std_logic := '0';
    signal cl_done   : std_logic;
    signal cl_class  : integer range 0 to N_CLS - 1;

    -- =========================================================================
    -- 7-segment display
    -- =========================================================================
    signal seg_r     : std_logic_vector(6 downto 0) := "1111111";
    signal an_r      : std_logic_vector(3 downto 0) := "1111";

begin

    -- =========================================================================
    -- MMCM: 100 MHz input -> 20 MHz output
    -- =========================================================================
    u_mmcm : MMCME2_BASE
        generic map (
            BANDWIDTH        => "OPTIMIZED",
            CLKFBOUT_MULT_F  => 10.0,    -- VCO = 100 * 10 = 1000 MHz
            CLKIN1_PERIOD    => 10.0,    -- 100 MHz input period (ns)
            CLKOUT0_DIVIDE_F => 50.0,    -- 1000 / 50 = 20 MHz
            DIVCLK_DIVIDE    => 1,
            CLKFBOUT_PHASE   => 0.0,
            CLKOUT0_DUTY_CYCLE => 0.5,
            CLKOUT0_PHASE    => 0.0,
            REF_JITTER1      => 0.01,
            STARTUP_WAIT     => FALSE
        )
        port map (
            CLKIN1   => clk,          -- 100 MHz from board oscillator
            CLKFBIN  => mmcm_fb,
            CLKFBOUT => mmcm_fb,
            CLKOUT0  => clk20_raw,
            LOCKED   => mmcm_locked,
            RST      => '0',
            PWRDWN   => '0'
        );

    u_clk20_buf : BUFG
        port map (I => clk20_raw, O => clk20);

    uart_txd <= uart_txd_r;

    -- =========================================================================
    -- Reset synchroniser: BTNC (active-high) or MMCM not-locked -> rstn='0'
    -- All downstream logic clocked on clk20.
    -- =========================================================================
    p_reset : process(clk20) is
    begin
        if rising_edge(clk20) then
            rst_meta <= btnc or not mmcm_locked;
            rst_sync <= rst_meta;
        end if;
    end process p_reset;

    rstn <= not rst_sync;

    -- =========================================================================
    -- UART-RX  (115200 baud, 8N1, LSB-first)
    -- =========================================================================
    p_uart : process(clk20) is
    begin
        if rising_edge(clk20) then
            uart_valid <= '0';
            if rstn = '0' then
                uart_state   <= U_IDLE;
                uart_clk_cnt <= 0;
                uart_bit_cnt <= 0;
            else
                case uart_state is

                    -- Wait for falling edge of start bit
                    when U_IDLE =>
                        if uart_rxd = '0' then
                            uart_clk_cnt <= 0;
                            uart_state   <= U_START;
                        end if;

                    -- Wait half-bit to land at centre of start bit,
                    -- then move to data sampling
                    when U_START =>
                        if uart_clk_cnt = CLKS_PER_BIT / 2 - 1 then
                            uart_clk_cnt <= 0;
                            uart_bit_cnt <= 0;
                            uart_state   <= U_DATA;
                        else
                            uart_clk_cnt <= uart_clk_cnt + 1;
                        end if;

                    -- Sample 8 data bits at centre of each bit period
                    when U_DATA =>
                        if uart_clk_cnt = CLKS_PER_BIT - 1 then
                            uart_clk_cnt <= 0;
                            -- Shift right: new bit -> MSB; LSB-first UART gives
                            -- correct byte after 8 shifts  (bit7..bit0 at [7:0])
                            uart_shift <= uart_rxd & uart_shift(7 downto 1);
                            if uart_bit_cnt = 7 then
                                uart_state <= U_STOP;
                            else
                                uart_bit_cnt <= uart_bit_cnt + 1;
                            end if;
                        else
                            uart_clk_cnt <= uart_clk_cnt + 1;
                        end if;

                    -- Wait one bit period for stop bit, then output byte
                    when U_STOP =>
                        if uart_clk_cnt = CLKS_PER_BIT - 1 then
                            uart_clk_cnt <= 0;
                            uart_data    <= uart_shift;
                            uart_valid   <= '1';
                            uart_state   <= U_IDLE;
                        else
                            uart_clk_cnt <= uart_clk_cnt + 1;
                        end if;

                end case;
            end if;
        end if;
    end process p_uart;

    -- =========================================================================
    -- UART-TX  (send one response byte after inference)
    -- =========================================================================
    p_uart_tx : process(clk20) is
    begin
        if rising_edge(clk20) then
            if rstn = '0' then
                uart_tx_state   <= TX_IDLE;
                uart_tx_clk_cnt <= 0;
                uart_tx_bit_cnt <= 0;
                uart_tx_shift   <= (others => '1');
                uart_tx_busy    <= '0';
                uart_txd_r      <= '1';
            else
                case uart_tx_state is

                    when TX_IDLE =>
                        uart_tx_busy <= '0';
                        uart_txd_r   <= '1';
                        if uart_tx_req = '1' then
                            uart_tx_shift   <= uart_tx_data;
                            uart_tx_clk_cnt <= 0;
                            uart_tx_bit_cnt <= 0;
                            uart_tx_state   <= TX_START;
                            uart_txd_r      <= '0';
                            uart_tx_busy    <= '1';
                        end if;

                    when TX_START =>
                        uart_tx_busy <= '1';
                        if uart_tx_clk_cnt = CLKS_PER_BIT - 1 then
                            uart_tx_clk_cnt <= 0;
                            uart_txd_r      <= uart_tx_shift(0);
                            uart_tx_state   <= TX_DATA;
                        else
                            uart_tx_clk_cnt <= uart_tx_clk_cnt + 1;
                        end if;

                    when TX_DATA =>
                        uart_tx_busy <= '1';
                        if uart_tx_clk_cnt = CLKS_PER_BIT - 1 then
                            uart_tx_clk_cnt <= 0;
                            if uart_tx_bit_cnt = 7 then
                                uart_txd_r    <= '1';
                                uart_tx_state <= TX_STOP;
                            else
                                uart_tx_shift   <= '0' & uart_tx_shift(7 downto 1);
                                uart_txd_r      <= uart_tx_shift(1);
                                uart_tx_bit_cnt <= uart_tx_bit_cnt + 1;
                            end if;
                        else
                            uart_tx_clk_cnt <= uart_tx_clk_cnt + 1;
                        end if;

                    when TX_STOP =>
                        uart_tx_busy <= '1';
                        if uart_tx_clk_cnt = CLKS_PER_BIT - 1 then
                            uart_tx_clk_cnt <= 0;
                            uart_tx_state   <= TX_IDLE;
                            uart_tx_busy    <= '0';
                            uart_txd_r      <= '1';
                        else
                            uart_tx_clk_cnt <= uart_tx_clk_cnt + 1;
                        end if;

                end case;
            end if;
        end if;
    end process p_uart_tx;

    -- =========================================================================
    -- Top-level control FSM
    --
    -- Timing for pixel-0:
    --   Cycle N (TS_IDLE, uart_valid='1'):
    --     first_px <- NORM_LUT[uart_data]
    --     pe_start = '1', cl_start = '1'
    --     -> TS_START_PULSE
    --   Cycle N+1 (TS_START_PULSE):
    --     patch_embed is now in ST_COLLECT (start was '1' on N)
    --     pe_i_valid = '1', pe_i_data = first_px   (pixel 0 accepted)
    --     -> TS_COLLECT, px_cnt = 1
    --   Cycles N+2 .. (uart gives bytes 1..783):
    --     Each uart_valid feeds one normalised pixel to patch_embed
    --   After px_cnt = N_PX-1 (pixel 783): -> TS_INFER
    -- =========================================================================
    p_ctrl : process(clk20) is
    begin
        if rising_edge(clk20) then
            -- Default: deassert strobes (overridden below)
            pe_start   <= '0';
            pe_i_valid <= '0';
            cl_start   <= '0';
            uart_tx_req <= '0';

            if rstn = '0' then
                top_state <= TS_IDLE;
                px_cnt    <= 0;
                result_reg <= 0;
                result_sent <= '0';
            else
                case top_state is

                    -- -------------------------------------------------------
                    when TS_IDLE =>
                        if uart_valid = '1' then
                            first_px  <= NORM_LUT(to_integer(unsigned(uart_data)));
                            pe_start  <= '1';
                            cl_start  <= '1';
                            px_cnt    <= 1;
                            top_state <= TS_START_PULSE;
                        end if;

                    -- -------------------------------------------------------
                    -- patch_embed is now in ST_COLLECT; present pixel 0
                    when TS_START_PULSE =>
                        pe_i_data  <= first_px;
                        pe_i_valid <= '1';
                        top_state  <= TS_COLLECT;

                    -- -------------------------------------------------------
                    -- Feed pixels 1..783 as UART bytes arrive
                    when TS_COLLECT =>
                        if uart_valid = '1' then
                            pe_i_data  <= NORM_LUT(to_integer(unsigned(uart_data)));
                            pe_i_valid <= '1';
                            if px_cnt = N_PX - 1 then
                                -- Probe ACK: let the PC know the full frame arrived
                                -- before the classifier finishes.
                                uart_tx_data <= x"A5";
                                uart_tx_req  <= '1';
                                top_state <= TS_INFER;
                            else
                                px_cnt <= px_cnt + 1;
                            end if;
                        end if;

                    -- -------------------------------------------------------
                    -- patch_embed -> encoder_block -> classifier running
                    when TS_INFER =>
                        if cl_done = '1' then
                            result_reg <= cl_class;
                            top_state  <= TS_SHOW;
                            result_sent <= '0';
                        end if;

                    -- -------------------------------------------------------
                    -- Result shown; send result byte; re-arm on next image's
                    -- first UART byte so no physical reset is needed.
                    when TS_SHOW =>
                        if (result_sent = '0') and (uart_tx_busy = '0') then
                            uart_tx_data <= std_logic_vector(to_unsigned(result_reg, 8));
                            uart_tx_req  <= '1';
                            result_sent  <= '1';
                        end if;

                        -- Auto re-arm: first byte of the NEXT image restarts pipeline
                        if uart_valid = '1' then
                            first_px    <= NORM_LUT(to_integer(unsigned(uart_data)));
                            pe_start    <= '1';
                            cl_start    <= '1';
                            px_cnt      <= 1;
                            result_sent <= '0';
                            top_state   <= TS_START_PULSE;
                        end if;

                end case;
            end if;
        end if;
    end process p_ctrl;

    result_cls <= result_reg;

    -- =========================================================================
    -- patch_embed
    -- =========================================================================
    u_patch_embed : entity work.patch_embed
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            PATCH_SIZE => PATCH_SIZE,
            IMG_SIZE   => IMG_SIZE,
            D_MODEL    => MODEL_DIM,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk   => clk20,
            rstn  => rstn,
            start => pe_start,
            done  => pe_done,
            i_data    => pe_i_data,
            i_valid   => pe_i_valid,
            o_data    => pe_o_data,
            o_valid   => pe_o_valid,
            o_last    => pe_o_last,
            o_channel => pe_o_channel
        );

    -- =========================================================================
    -- encoder_block  (structural architecture -- avoids ieee.math_real runtime)
    -- =========================================================================
    u_encoder : entity work.encoder_block(structural)
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            MODEL_DIM  => MODEL_DIM,
            NUM_HEADS  => NUM_HEADS,
            HEAD_DIM   => HEAD_DIM,
            HIDDEN_DIM => HIDDEN_DIM,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk  => clk20,
            rstn => rstn,

            i_data    => pe_o_data,
            i_valid   => pe_o_valid,
            i_last    => pe_o_last,
            i_channel => pe_o_channel,

            o_data    => enc_o_data,
            o_valid   => enc_o_valid,
            o_last    => enc_o_last,
            o_channel => enc_o_channel,

            -- Debug taps (unused)
            o_mha_data    => enc_mha_data,
            o_mha_valid   => enc_mha_valid,
            o_mha_last    => enc_mha_last,
            o_mha_channel => enc_mha_channel,
            o_ffn_data    => enc_ffn_data,
            o_ffn_valid   => enc_ffn_valid,
            o_ffn_last    => enc_ffn_last,
            o_ffn_channel => enc_ffn_channel,

            -- MHA weight memory
            w_q_addr => w_q_addr_s, w_q_data => w_q_data_s, w_q_re => w_q_re_s,
            w_k_addr => w_k_addr_s, w_k_data => w_k_data_s, w_k_re => w_k_re_s,
            w_v_addr => w_v_addr_s, w_v_data => w_v_data_s, w_v_re => w_v_re_s,
            w_o_addr => w_o_addr_s, w_o_data => w_o_data_s, w_o_re => w_o_re_s,

            -- FFN weight/bias memory
            ffn_w1_addr => ffn_w1_addr_s, ffn_w1_data => ffn_w1_data_s, ffn_w1_re => ffn_w1_re_s,
            ffn_b1_addr => ffn_b1_addr_s, ffn_b1_data => ffn_b1_data_s, ffn_b1_re => ffn_b1_re_s,
            ffn_w2_addr => ffn_w2_addr_s, ffn_w2_data => ffn_w2_data_s, ffn_w2_re => ffn_w2_re_s,
            ffn_b2_addr => ffn_b2_addr_s, ffn_b2_data => ffn_b2_data_s, ffn_b2_re => ffn_b2_re_s,

            -- LayerNorm parameters: not loaded (LN uses vanilla norm, no learnable scale)
            ln_params_data  => (others => '0'),
            ln_params_valid => '0',
            ln_params_addr  => (others => '0'),
            ln_params_sel   => '0'
        );

    -- =========================================================================
    -- weight_mem  (encoder BRAM ROMs from weights_pkg)
    -- =========================================================================
    u_weight_mem : entity work.weight_mem
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            MODEL_DIM  => MODEL_DIM,
            HIDDEN_DIM => HIDDEN_DIM
        )
        port map (
            clk => clk20,

            w_q_addr => w_q_addr_s, w_q_re => w_q_re_s, w_q_data => w_q_data_s,
            w_k_addr => w_k_addr_s, w_k_re => w_k_re_s, w_k_data => w_k_data_s,
            w_v_addr => w_v_addr_s, w_v_re => w_v_re_s, w_v_data => w_v_data_s,
            w_o_addr => w_o_addr_s, w_o_re => w_o_re_s, w_o_data => w_o_data_s,

            w1_addr  => ffn_w1_addr_s, w1_re => ffn_w1_re_s, w1_rdata => ffn_w1_data_s,
            b1_addr  => ffn_b1_addr_s, b1_re => ffn_b1_re_s, b1_rdata => ffn_b1_data_s,
            w2_addr  => ffn_w2_addr_s, w2_re => ffn_w2_re_s, w2_rdata => ffn_w2_data_s,
            b2_addr  => ffn_b2_addr_s, b2_re => ffn_b2_re_s, b2_rdata => ffn_b2_data_s
        );

    -- =========================================================================
    -- classifier
    -- =========================================================================
    u_classifier : entity work.classifier
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            D_MODEL    => MODEL_DIM,
            SEQ_LEN    => SEQ_LEN,
            N_CLS      => N_CLS
        )
        port map (
            clk   => clk20,
            rstn  => rstn,
            start => cl_start,
            done  => cl_done,
            i_data    => enc_o_data,
            i_valid   => enc_o_valid,
            i_last    => enc_o_last,
            i_channel => enc_o_channel,
            o_class   => cl_class
        );

    -- =========================================================================
    -- 7-segment display
    --   seg[6:0] = {g,f,e,d,c,b,a}, active-low
    --   Digit shown on an[0] (rightmost) when result is ready
    -- =========================================================================
    p_seg : process(clk20) is
    begin
        if rising_edge(clk20) then
            if rstn = '0' then
                seg_r <= "1111111";  -- all segments off
                an_r  <= "1111";    -- all digits off
            else
                -- Decode result class to 7-segment pattern
                case result_cls is
                    when 0      => seg_r <= "1000000";   --  _
                    when 1      => seg_r <= "1111001";   --   |  |
                    when 2      => seg_r <= "0100100";   -- _|  |_
                    when 3      => seg_r <= "0110000";
                    when 4      => seg_r <= "0011001";
                    when 5      => seg_r <= "0010010";
                    when 6      => seg_r <= "0000010";
                    when 7      => seg_r <= "1111000";
                    when 8      => seg_r <= "0000000";
                    when 9      => seg_r <= "0010000";
                    when others => seg_r <= "1111111";
                end case;

                -- Enable digit 0 only when result is available
                if top_state = TS_SHOW then
                    an_r <= "1110";   -- an[0] active, others off
                else
                    an_r <= "1111";   -- all off while processing
                end if;
            end if;
        end if;
    end process p_seg;

    seg <= seg_r;
    dp  <= '1';    -- decimal point always off
    an  <= an_r;

    -- =========================================================================
    -- Status LEDs
    --   led[0] : pipeline active (not idle)
    --   led[1] : receiving pixels
    --   led[2] : inference in progress
    --   led[3] : result ready
    -- =========================================================================
    led(0) <= '0' when top_state = TS_IDLE  else '1';
    led(1) <= '1' when top_state = TS_COLLECT or top_state = TS_START_PULSE
                  else '0';
    led(2) <= '1' when top_state = TS_INFER  else '0';
    led(3) <= '1' when top_state = TS_SHOW   else '0';

end architecture rtl;

-- =============================================================================
-- End of basys3_top.vhd
-- =============================================================================
