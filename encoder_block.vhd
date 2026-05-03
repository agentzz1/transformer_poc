-- ============================================================================
-- encoder_block.vhd -- Post-LN Transformer Encoder Block (Top-Level Structural)
-- ============================================================================
-- Dataflow:
--   i_data -> mha_controller -> residual_add_1 -> ffn -> residual_add_2 -> o_data
--   i_data --------> (input_buffer skip) ------^                          ^
--   residual_add_1 out -> (res1_buffer skip) ------------------------------|
--
-- Post-LN: each residual_add performs element-wise addition of main-path
-- and skip-connection streams, then applies LayerNorm to the sum.
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.utilities.all;
use work.clog2_pkg.all;

entity encoder_block is
    generic (
        DATA_WIDTH : positive := 16;
        MODEL_DIM  : positive := 512;
        NUM_HEADS  : positive := 8;
        HEAD_DIM   : positive := 64;
        HIDDEN_DIM : positive := 2048;
        SEQ_LEN    : positive := 64
    );
    port (
        -- Clock and reset
        clk   : in  std_logic;
        rstn  : in  std_logic;

        -- Streaming input
        i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid   : in  std_logic;
        i_last    : in  std_logic;
        i_channel : in  integer range 0 to max_size_x - 1;

        -- Streaming output
        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer range 0 to max_size_x - 1;

        -- MHA sublayer debug tap
        o_mha_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_mha_valid   : out std_logic;
        o_mha_last    : out std_logic;
        o_mha_channel : out integer range 0 to max_size_x - 1;

        -- FFN sublayer debug tap
        o_ffn_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_ffn_valid   : out std_logic;
        o_ffn_last    : out std_logic;
        o_ffn_channel : out integer range 0 to max_size_x - 1;

        -- MHA weight memory interfaces (pass-through to mha_controller)
        w_q_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_q_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_q_re   : out std_logic;
        w_k_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_k_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_k_re   : out std_logic;
        w_v_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_v_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_v_re   : out std_logic;
        w_o_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_o_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_o_re   : out std_logic;

        -- FFN weight / bias memory interfaces (pass-through to ffn)
        ffn_w1_addr  : out std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
        ffn_w1_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_w1_re    : out std_logic;
        ffn_b1_addr  : out std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
        ffn_b1_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_b1_re    : out std_logic;
        ffn_w2_addr  : out std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
        ffn_w2_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_w2_re    : out std_logic;
        ffn_b2_addr  : out std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
        ffn_b2_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_b2_re    : out std_logic;

        -- LayerNorm parameter loading (pass-through to residual_add instances)
        ln_params_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ln_params_valid : in  std_logic;
        ln_params_addr  : in  std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
        ln_params_sel   : in  std_logic  -- '0' = residual_add_1, '1' = residual_add_2
    );
end entity encoder_block;

architecture structural of encoder_block is

    -- =========================================================================
    -- Constants
    -- =========================================================================
    constant MODEL_DIM_BITS : positive := clog2(MODEL_DIM);

    -- =========================================================================
    -- Component declarations
    -- =========================================================================

    component mha_controller is
        generic (
            DATA_WIDTH : positive := 16;
            MODEL_DIM  : positive := 512;
            NUM_HEADS  : positive := 8;
            HEAD_DIM   : positive := 64;
            SEQ_LEN    : positive := 64
        );
        port (
            clk       : in  std_logic;
            rstn      : in  std_logic;
            i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_valid   : in  std_logic;
            i_last    : in  std_logic;
            i_channel : in  integer range 0 to max_size_x - 1;
            o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid   : out std_logic;
            o_last    : out std_logic;
            o_channel : out integer range 0 to max_size_x - 1;
            start     : in  std_logic;
            mode      : in  std_logic_vector(1 downto 0);
            done      : out std_logic;
            -- Weight memory
            w_q_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_q_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_q_re    : out std_logic;
            w_k_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_k_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_k_re    : out std_logic;
            w_v_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_v_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_v_re    : out std_logic;
            w_o_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_o_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_o_re    : out std_logic
        );
    end component mha_controller;

    component residual_add is
        generic (
            DATA_WIDTH : positive := 16;
            VEC_SIZE   : positive := 512
        );
        port (
            clk                : in  std_logic;
            rstn               : in  std_logic;
            i_data             : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_data_valid       : in  std_logic;
            i_data_last        : in  std_logic;
            i_data_channel     : in  integer range 0 to max_size_x - 1;
            i_residual         : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_residual_valid   : in  std_logic;
            i_residual_last    : in  std_logic;
            i_residual_channel : in  integer range 0 to max_size_x - 1;
            o_data             : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid            : out std_logic;
            o_last             : out std_logic;
            o_channel          : out integer range 0 to max_size_x - 1;
            -- LayerNorm parameter loading
            i_params_data      : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_params_valid     : in  std_logic;
            i_params_addr      : in  std_logic_vector(clog2(VEC_SIZE) - 1 downto 0);
            i_params_sel       : in  std_logic  -- '0'=gamma, '1'=beta
        );
    end component residual_add;

    component ffn is
        generic (
            DATA_WIDTH : positive := 16;
            MODEL_DIM  : positive := 512;
            HIDDEN_DIM : positive := 2048
        );
        port (
            clk       : in  std_logic;
            rstn      : in  std_logic;
            i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_valid   : in  std_logic;
            i_last    : in  std_logic;
            i_channel : in  integer range 0 to max_size_x - 1;
            o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid   : out std_logic;
            o_last    : out std_logic;
            o_channel : out integer range 0 to max_size_x - 1;
            start     : in  std_logic;
            done      : out std_logic;
            -- Weight / bias memory
            w1_addr   : out std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
            w1_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w1_re     : out std_logic;
            b1_addr   : out std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
            b1_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            b1_re     : out std_logic;
            w2_addr   : out std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
            w2_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w2_re     : out std_logic;
            b2_addr   : out std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
            b2_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            b2_re     : out std_logic
        );
    end component ffn;

    component control_unit is
        generic (
            DATA_WIDTH : positive := 16;
            SEQ_LEN    : positive := 64
        );
        port (
            clk                : in  std_logic;
            rstn               : in  std_logic;
            i_valid            : in  std_logic;
            i_ready            : out std_logic;
            o_ready            : in  std_logic;
            o_valid            : out std_logic;
            mha_start          : out std_logic;
            mha_mode           : out std_logic_vector(1 downto 0);
            ffn_start          : out std_logic;
            residual_add_1_en  : out std_logic;
            residual_add_2_en  : out std_logic;
            layernorm_1_en     : out std_logic;
            layernorm_2_en     : out std_logic;
            mha_done           : in  std_logic;
            ffn_done           : in  std_logic;
            residual_add_1_done: in  std_logic;
            residual_add_2_done: in  std_logic;
            layernorm_1_done   : in  std_logic;
            layernorm_2_done   : in  std_logic;
            seq_pos            : out unsigned(integer(ceil(log2(real(SEQ_LEN)))) downto 0)
        );
    end component control_unit;

    -- =========================================================================
    -- Internal data-path signals
    -- =========================================================================

    -- MHA sublayer output stream
    signal mha_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal mha_out_valid   : std_logic;
    signal mha_out_last    : std_logic;
    signal mha_out_channel : integer range 0 to max_size_x - 1;

    -- First residual-add + LN output stream (input to FFN)
    signal res1_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal res1_out_valid   : std_logic;
    signal res1_out_last    : std_logic;
    signal res1_out_channel : integer range 0 to max_size_x - 1;

    -- FFN sublayer output stream
    signal ffn_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_out_valid   : std_logic;
    signal ffn_out_last    : std_logic;
    signal ffn_out_channel : integer range 0 to max_size_x - 1;

    -- Second residual-add + LN output stream (final encoder output)
    signal res2_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal res2_out_valid   : std_logic;
    signal res2_out_last    : std_logic;
    signal res2_out_channel : integer range 0 to max_size_x - 1;

    -- =========================================================================
    -- Input buffer — captures i_data for skip connection to residual_add_1
    -- =========================================================================
    type token_buffer_t is array (0 to MODEL_DIM - 1) of
        std_logic_vector(DATA_WIDTH - 1 downto 0);

    signal input_buffer       : token_buffer_t;
    signal input_buf_write    : std_logic;
    signal input_buf_waddr    : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal input_buf_channel  : integer range 0 to max_size_x - 1;
    signal input_buf_full     : std_logic;

    -- Replay state for input buffer -> residual_add_1 skip
    signal replay1_data       : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal replay1_valid      : std_logic;
    signal replay1_last       : std_logic;
    signal replay1_channel    : integer range 0 to max_size_x - 1;
    signal replay1_raddr      : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal replay1_active     : std_logic;

    -- =========================================================================
    -- Res1 buffer — captures residual_add_1 output for skip to residual_add_2
    -- =========================================================================
    signal res1_buffer        : token_buffer_t;
    signal res1_buf_write     : std_logic;
    signal res1_buf_waddr     : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal res1_buf_channel   : integer range 0 to max_size_x - 1;
    signal res1_buf_full      : std_logic;

    -- Replay state for res1 buffer -> residual_add_2 skip
    signal replay2_data       : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal replay2_valid      : std_logic;
    signal replay2_last       : std_logic;
    signal replay2_channel    : integer range 0 to max_size_x - 1;
    signal replay2_raddr      : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal replay2_active     : std_logic;

    -- =========================================================================
    -- Control-unit interface signals
    -- =========================================================================
    signal cu_i_ready             : std_logic;
    signal cu_o_ready             : std_logic;
    signal cu_o_valid             : std_logic;

    signal mha_start              : std_logic;
    signal mha_mode               : std_logic_vector(1 downto 0);
    signal ffn_start              : std_logic;
    signal residual_add_1_en      : std_logic;
    signal residual_add_2_en      : std_logic;
    signal layernorm_1_en         : std_logic;
    signal layernorm_2_en         : std_logic;

    signal mha_done               : std_logic;
    signal ffn_done               : std_logic;
    signal residual_add_1_done    : std_logic;
    signal residual_add_2_done    : std_logic;
    signal layernorm_1_done       : std_logic;
    signal layernorm_2_done       : std_logic;

    signal seq_pos                : unsigned(integer(ceil(log2(real(SEQ_LEN)))) downto 0);

    -- =========================================================================
    -- Element counters for done-signal generation
    -- =========================================================================
    signal mha_in_cnt             : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal mha_out_elem_cnt       : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal res1_out_elem_cnt      : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal ffn_out_elem_cnt       : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal res2_out_elem_cnt      : unsigned(MODEL_DIM_BITS - 1 downto 0);

    -- Edge detectors for done signals (one-cycle pulse generation)
    signal mha_out_last_d1        : std_logic;
    signal res1_out_last_d1       : std_logic;
    signal ffn_out_last_d1        : std_logic;
    signal res2_out_last_d1       : std_logic;

begin

    -- =========================================================================
    -- Control unit instance
    -- =========================================================================
    u_control_unit : control_unit
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk                 => clk,
            rstn                => rstn,
            i_valid             => i_valid,
            i_ready             => cu_i_ready,
            o_ready             => cu_o_ready,
            o_valid             => cu_o_valid,
            mha_start           => mha_start,
            mha_mode            => mha_mode,
            ffn_start           => ffn_start,
            residual_add_1_en   => residual_add_1_en,
            residual_add_2_en   => residual_add_2_en,
            layernorm_1_en      => layernorm_1_en,
            layernorm_2_en      => layernorm_2_en,
            mha_done            => mha_done,
            ffn_done            => ffn_done,
            residual_add_1_done => residual_add_1_done,
            residual_add_2_done => residual_add_2_done,
            layernorm_1_done    => layernorm_1_done,
            layernorm_2_done    => layernorm_2_done,
            seq_pos             => seq_pos
        );

    -- Downstream is always ready (no backpressure at encoder-block boundary)
    cu_o_ready <= '1';

    -- =========================================================================
    -- MHA controller instance
    -- =========================================================================
    u_mha_controller : mha_controller
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            MODEL_DIM  => MODEL_DIM,
            NUM_HEADS  => NUM_HEADS,
            HEAD_DIM   => HEAD_DIM,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            i_data    => i_data,
            i_valid   => i_valid,
            i_last    => i_last,
            i_channel => i_channel,
            o_data    => mha_out_data,
            o_valid   => mha_out_valid,
            o_last    => mha_out_last,
            o_channel => mha_out_channel,
            start     => mha_start,
            mode      => mha_mode,
            done      => mha_done,
            w_q_addr  => w_q_addr,
            w_q_data  => w_q_data,
            w_q_re    => w_q_re,
            w_k_addr  => w_k_addr,
            w_k_data  => w_k_data,
            w_k_re    => w_k_re,
            w_v_addr  => w_v_addr,
            w_v_data  => w_v_data,
            w_v_re    => w_v_re,
            w_o_addr  => w_o_addr,
            w_o_data  => w_o_data,
            w_o_re    => w_o_re
        );

    -- =========================================================================
    -- Input buffer: capture i_data stream for skip connection to residual_add_1
    --
    -- Writes every valid input element into the buffer.  The write-address
    -- resets on i_last (end of token) or when the buffer is replayed.
    -- =========================================================================
    input_buf_write <= i_valid and cu_i_ready;

    p_input_buffer : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                input_buf_waddr   <= (others => '0');
                input_buf_channel <= 0;
                input_buf_full    <= '0';
            else
                if input_buf_write = '1' then
                    input_buffer(to_integer(input_buf_waddr)) <= i_data;
                    input_buf_channel <= i_channel;

                    if i_last = '1' then
                        input_buf_waddr <= (others => '0');
                        input_buf_full  <= '1';
                    else
                        input_buf_waddr <= input_buf_waddr + 1;
                    end if;
                end if;

                -- Clear full flag at start of replay
                if residual_add_1_en = '1' then
                    input_buf_full <= '0';
                end if;
            end if;
        end if;
    end process p_input_buffer;

    -- =========================================================================
    -- Replay 1: stream input_buffer out as residual_add_1 skip connection
    --
    -- Activated by residual_add_1_en.  Reads from input_buffer in lockstep
    -- with the MHA output stream so both arrive at residual_add_1 together.
    -- =========================================================================
    p_replay1_ctl : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                replay1_active <= '0';
                replay1_raddr  <= (others => '0');
            else
                if residual_add_1_en = '1' and input_buf_full = '1' then
                    replay1_active <= '1';
                    replay1_raddr  <= (others => '0');
                elsif replay1_active = '1' and mha_out_valid = '1' and mha_out_last = '1' then
                    replay1_active <= '0';
                    replay1_raddr  <= (others => '0');
                elsif replay1_active = '1' and mha_out_valid = '1' then
                    replay1_raddr <= replay1_raddr + 1;
                end if;
            end if;
        end if;
    end process p_replay1_ctl;

    replay1_data    <= input_buffer(to_integer(replay1_raddr));
    replay1_valid   <= replay1_active and mha_out_valid;
    replay1_last    <= mha_out_last when replay1_active = '1' else '0';
    replay1_channel <= input_buf_channel;

    -- =========================================================================
    -- First residual-add + LayerNorm instance
    --
    -- Main path: MHA controller output
    -- Skip path: replayed original input
    -- =========================================================================
    u_residual_add_1 : residual_add
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            VEC_SIZE   => MODEL_DIM
        )
        port map (
            clk                => clk,
            rstn               => rstn,
            i_data             => mha_out_data,
            i_data_valid       => mha_out_valid,
            i_data_last        => mha_out_last,
            i_data_channel     => mha_out_channel,
            i_residual         => replay1_data,
            i_residual_valid   => replay1_valid,
            i_residual_last    => replay1_last,
            i_residual_channel => replay1_channel,
            o_data             => res1_out_data,
            o_valid            => res1_out_valid,
            o_last             => res1_out_last,
            o_channel          => res1_out_channel,
            i_params_data      => ln_params_data,
            i_params_valid     => ln_params_valid,
            i_params_addr      => ln_params_addr,
            i_params_sel       => '0'
        );

    -- =========================================================================
    -- Res1 buffer: capture residual_add_1 output for skip to residual_add_2
    -- =========================================================================
    res1_buf_write <= res1_out_valid;

    p_res1_buffer : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                res1_buf_waddr   <= (others => '0');
                res1_buf_channel <= 0;
                res1_buf_full    <= '0';
            else
                if res1_buf_write = '1' then
                    res1_buffer(to_integer(res1_buf_waddr)) <= res1_out_data;
                    res1_buf_channel <= res1_out_channel;

                    if res1_out_last = '1' then
                        res1_buf_waddr <= (others => '0');
                        res1_buf_full  <= '1';
                    else
                        res1_buf_waddr <= res1_buf_waddr + 1;
                    end if;
                end if;

                if residual_add_2_en = '1' then
                    res1_buf_full <= '0';
                end if;
            end if;
        end if;
    end process p_res1_buffer;

    -- =========================================================================
    -- FFN instance
    -- =========================================================================
    u_ffn : ffn
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            MODEL_DIM  => MODEL_DIM,
            HIDDEN_DIM => HIDDEN_DIM
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            i_data    => res1_out_data,
            i_valid   => res1_out_valid,
            i_last    => res1_out_last,
            i_channel => res1_out_channel,
            o_data    => ffn_out_data,
            o_valid   => ffn_out_valid,
            o_last    => ffn_out_last,
            o_channel => ffn_out_channel,
            start     => ffn_start,
            done      => ffn_done,
            w1_addr   => ffn_w1_addr,
            w1_rdata  => ffn_w1_data,
            w1_re     => ffn_w1_re,
            b1_addr   => ffn_b1_addr,
            b1_rdata  => ffn_b1_data,
            b1_re     => ffn_b1_re,
            w2_addr   => ffn_w2_addr,
            w2_rdata  => ffn_w2_data,
            w2_re     => ffn_w2_re,
            b2_addr   => ffn_b2_addr,
            b2_rdata  => ffn_b2_data,
            b2_re     => ffn_b2_re
        );

    -- =========================================================================
    -- Replay 2: stream res1_buffer out as residual_add_2 skip connection
    -- =========================================================================
    p_replay2_ctl : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                replay2_active <= '0';
                replay2_raddr  <= (others => '0');
            else
                if residual_add_2_en = '1' and res1_buf_full = '1' then
                    replay2_active <= '1';
                    replay2_raddr  <= (others => '0');
                elsif replay2_active = '1' and ffn_out_valid = '1' and ffn_out_last = '1' then
                    replay2_active <= '0';
                    replay2_raddr  <= (others => '0');
                elsif replay2_active = '1' and ffn_out_valid = '1' then
                    replay2_raddr <= replay2_raddr + 1;
                end if;
            end if;
        end if;
    end process p_replay2_ctl;

    replay2_data    <= res1_buffer(to_integer(replay2_raddr));
    replay2_valid   <= replay2_active and ffn_out_valid;
    replay2_last    <= ffn_out_last when replay2_active = '1' else '0';
    replay2_channel <= res1_buf_channel;

    -- =========================================================================
    -- Second residual-add + LayerNorm instance
    --
    -- Main path: FFN output
    -- Skip path: replayed residual_add_1 output
    -- =========================================================================
    u_residual_add_2 : residual_add
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            VEC_SIZE   => MODEL_DIM
        )
        port map (
            clk                => clk,
            rstn               => rstn,
            i_data             => ffn_out_data,
            i_data_valid       => ffn_out_valid,
            i_data_last        => ffn_out_last,
            i_data_channel     => ffn_out_channel,
            i_residual         => replay2_data,
            i_residual_valid   => replay2_valid,
            i_residual_last    => replay2_last,
            i_residual_channel => replay2_channel,
            o_data             => res2_out_data,
            o_valid            => res2_out_valid,
            o_last             => res2_out_last,
            o_channel          => res2_out_channel,
            i_params_data      => ln_params_data,
            i_params_valid     => ln_params_valid,
            i_params_addr      => ln_params_addr,
            i_params_sel       => '1'
        );

    -- =========================================================================
    -- Done-signal generation for control_unit
    --
    -- residual_add has no explicit "done" port — we detect completion from
    -- its output stream.  The internal FSM goes:
    --   IDLE -> BUFFER_RESIDUAL -> ADD_ELEMENTS -> LAYERNORM -> DONE
    --
    -- We generate:
    --   residual_add_N_done : addition phase complete (main + residual summed)
    --   layernorm_N_done    : LayerNorm output stream finished
    --
    -- For the encoder block, we detect both from the residual_add output:
    --   res_add_done pulses when the last element enters the add pipeline
    --   ln_done pulses when the last LN output element emerges
    -- =========================================================================

    -- Edge-detect the o_last signals to produce one-cycle pulses
    p_done_edge : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                mha_out_last_d1   <= '0';
                res1_out_last_d1  <= '0';
                ffn_out_last_d1   <= '0';
                res2_out_last_d1  <= '0';
            else
                mha_out_last_d1   <= mha_out_valid and mha_out_last;
                res1_out_last_d1  <= res1_out_valid and res1_out_last;
                ffn_out_last_d1   <= ffn_out_valid and ffn_out_last;
                res2_out_last_d1  <= res2_out_valid and res2_out_last;
            end if;
        end if;
    end process p_done_edge;

    -- residual_add_1_done: pulse when the last MHA output element has been
    -- consumed (addition phase finishes).  We detect this from the MHA output
    -- stream's last edge.
    residual_add_1_done <= '1' when (mha_out_valid = '1' and mha_out_last = '1')
                                    and residual_add_1_en = '1'
                           else '0';

    -- layernorm_1_done: pulse when residual_add_1 finishes its LN output stream
    layernorm_1_done <= '1' when (res1_out_valid = '1' and res1_out_last = '1')
                                  and res1_out_last_d1 = '0'
                        else '0';

    -- residual_add_2_done: pulse when the last FFN output element has been
    -- consumed (addition phase finishes).
    residual_add_2_done <= '1' when (ffn_out_valid = '1' and ffn_out_last = '1')
                                     and residual_add_2_en = '1'
                            else '0';

    -- layernorm_2_done: pulse when residual_add_2 finishes its LN output stream
    layernorm_2_done <= '1' when (res2_out_valid = '1' and res2_out_last = '1')
                                   and res2_out_last_d1 = '0'
                         else '0';

    -- =========================================================================
    -- Output assignments
    -- =========================================================================

    -- Final encoder output: pass-through from residual_add_2
    o_data    <= res2_out_data;
    o_valid   <= res2_out_valid;
    o_last    <= res2_out_last;
    o_channel <= res2_out_channel;

    -- MHA debug tap: connected to MHA controller output
    o_mha_data    <= mha_out_data;
    o_mha_valid   <= mha_out_valid;
    o_mha_last    <= mha_out_last;
    o_mha_channel <= mha_out_channel;

    -- FFN debug tap: connected to FFN output
    o_ffn_data    <= ffn_out_data;
    o_ffn_valid   <= ffn_out_valid;
    o_ffn_last    <= ffn_out_last;
    o_ffn_channel <= ffn_out_channel;

end architecture structural;

-- ============================================================================
-- End of file encoder_block.vhd
-- ============================================================================
