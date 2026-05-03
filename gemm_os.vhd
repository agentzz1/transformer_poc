--------------------------------------------------------------------------------
-- gemm_os.vhd — Output-Stationary GEMM Systolic Array
--
-- Post-LN Transformer encoder block MAC engine.
-- Dataflow: partial sums stay stationary in each PE,
--           weights flow horizontally (left -> right),
--           activations flow vertically (top -> bottom).
--
-- Operation (two-pass):
--   1. ACCUM  — K inner-dimension cycles of MAC.
--      Inputs must arrive pre-skewed, or internal skew buffers provide the
--      triangular delay needed for correct data alignment.
--   2. READOUT — channel-sequential streaming of accumulated results.
--
-- Interface matches accel library conventions:
--   valid / last handshaking, channel-indexed sequential output
--   (compatible with psum_activation / psum_requantize downstream consumers).
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

    use work.utilities.all;

--------------------------------------------------------------------------------
-- pe_os — Processing Element (output-stationary)
--
-- Two-stage pipeline:
--   Stage 0: register streaming inputs (weight from West, activation from North)
--   Stage 1: MAC  — multiply registered operands, accumulate into local psum
--
-- Passthrough: weight flows East, activation flows South (1 cycle delay each).
-- Valid propagates alongside data through the grid.
--------------------------------------------------------------------------------

entity pe_os is
    generic (
        DATA_WIDTH   : positive := 16;
        ACCUM_WIDTH  : positive := 32
    );
    port (
        clk      : in  std_logic;
        rstn     : in  std_logic;

        -- Streaming inputs (North / West edges)
        i_weight : in  signed(DATA_WIDTH - 1 downto 0);
        i_act    : in  signed(DATA_WIDTH - 1 downto 0);
        i_valid  : in  std_logic;

        -- Registered passthrough outputs (South / East edges)
        o_weight : out signed(DATA_WIDTH - 1 downto 0);
        o_act    : out signed(DATA_WIDTH - 1 downto 0);
        o_valid  : out std_logic;

        -- Accumulator control
        i_clear  : in  std_logic;

        -- Accumulated partial sum (read port, held stationary)
        o_psum   : out signed(ACCUM_WIDTH - 1 downto 0)
    );
end entity pe_os;

architecture rtl of pe_os is

    -- Stage 0 registers
    signal weight_r : signed(DATA_WIDTH - 1 downto 0);
    signal act_r    : signed(DATA_WIDTH - 1 downto 0);
    signal valid_r  : std_logic;

    -- Accumulator
    signal psum_r   : signed(ACCUM_WIDTH - 1 downto 0);

    -- Combinational product (sized for full-precision multiply)
    signal product  : signed(2 * DATA_WIDTH - 1 downto 0);

begin

    -- Combinational multiply on registered operands
    product <= weight_r * act_r;

    p_reg : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                weight_r <= (others => '0');
                act_r    <= (others => '0');
                valid_r  <= '0';
                psum_r   <= (others => '0');
            else
                -- Stage 0: capture streaming inputs
                weight_r <= i_weight;
                act_r    <= i_act;
                valid_r  <= i_valid;

                -- Stage 1: MAC using registered operands from the *previous* cycle.
                -- valid_r is the one-cycle-delayed version of i_valid, so the MAC
                -- runs exactly when both valid operands are present in weight_r/act_r.
                if valid_r = '1' then
                    psum_r <= psum_r + resize(product, ACCUM_WIDTH);
                end if;

                if i_clear = '1' then
                    psum_r <= (others => '0');
                end if;
            end if;
        end if;
    end process p_reg;

    -- Passthrough outputs (registered, one-cycle delay)
    o_weight <= weight_r;
    o_act    <= act_r;
    o_valid  <= valid_r;

    -- Accumulator read port
    o_psum <= psum_r;

end architecture rtl;


--------------------------------------------------------------------------------
-- gemm_os — Output-Stationary GEMM Systolic Array Top-Level
--------------------------------------------------------------------------------

entity gemm_os is
    generic (
        ROWS        : positive := 4;
        COLS        : positive := 4;
        DATA_WIDTH  : positive := 16;
        ACCUM_WIDTH : positive := 32
    );
    port (
        clk  : in std_logic;
        rstn : in std_logic;

        -- Weight inputs: flat vector, ROWS * DATA_WIDTH bits.
        -- One weight per row enters the left edge each cycle.
        -- Over K cycles the full ROWS x K weight matrix streams through.
        i_weight       : in  std_logic_vector(ROWS * DATA_WIDTH - 1 downto 0);
        i_weight_valid : in  std_logic;
        i_weight_last  : in  std_logic;

        -- Activation inputs: flat vector, COLS * DATA_WIDTH bits.
        -- One activation per column enters the top edge each cycle.
        -- Over K cycles the full K x COLS activation matrix streams through.
        i_act          : in  std_logic_vector(COLS * DATA_WIDTH - 1 downto 0);
        i_act_valid    : in  std_logic;
        i_act_last     : in  std_logic;

        -- Output stream: channel-sequential, accel-library compatible.
        o_data         : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid        : out std_logic;
        o_last         : out std_logic;
        o_channel      : out integer range 0 to max_size_x - 1
    );
end entity gemm_os;


architecture rtl of gemm_os is

    ---------------------------------------------------------------------------
    -- Type definitions
    ---------------------------------------------------------------------------
    subtype data_t  is signed(DATA_WIDTH - 1 downto 0);
    subtype accum_t is signed(ACCUM_WIDTH - 1 downto 0);

    -- 2-D grid of PE I/O wires: (row, col)
    type grid_data_t is array (0 to ROWS - 1, 0 to COLS - 1) of data_t;
    type grid_valid_t is array (0 to ROWS - 1, 0 to COLS - 1) of std_logic;
    type grid_accum_t is array (0 to ROWS - 1, 0 to COLS - 1) of accum_t;

    -- 1-D slice for input skew shift registers
    type skew_data_t is array (natural range <>) of data_t;
    type skew_valid_t is array (natural range <>) of std_logic;

    ---------------------------------------------------------------------------
    -- Input skew buffers
    --
    -- Without skew, PE(r,c) would pair W[r,k] with A[k+r-c,c] instead of
    -- A[k,c]. Skew registers add r cycles to each weight row and c cycles
    -- to each activation column, so W[r,k] and A[k,c] both reach PE(r,c)
    -- at cycle k + r + c.
    ---------------------------------------------------------------------------

    -- Per-row weight skew: depth = row index
    type row_skew_data_t is array (0 to ROWS - 1) of skew_data_t(0 to ROWS - 1);
    type row_skew_valid_t is array (0 to ROWS - 1) of skew_valid_t(0 to ROWS - 1);
    signal skew_w_data  : row_skew_data_t;
    signal skew_w_valid : row_skew_valid_t;

    -- Per-column activation skew: depth = col index
    type col_skew_data_t is array (0 to COLS - 1) of skew_data_t(0 to COLS - 1);
    type col_skew_valid_t is array (0 to COLS - 1) of skew_valid_t(0 to COLS - 1);
    signal skew_a_data  : col_skew_data_t;
    signal skew_a_valid : col_skew_valid_t;

    -- Skewed values at PE boundary (left edge for weights, top edge for activations)
    signal w_skewed : skew_data_t(0 to ROWS - 1);
    signal wv_skewed : skew_valid_t(0 to ROWS - 1);
    signal a_skewed : skew_data_t(0 to COLS - 1);
    signal av_skewed : skew_valid_t(0 to COLS - 1);

    ---------------------------------------------------------------------------
    -- PE grid I/O signals
    ---------------------------------------------------------------------------
    -- Weight flowing into PE(r,c) from the left
    signal w_in  : grid_data_t;
    signal wv_in : grid_valid_t;

    -- Activation flowing into PE(r,c) from the top
    signal a_in  : grid_data_t;
    signal av_in : grid_valid_t;

    -- Weight flowing out of PE(r,c) to the right
    signal w_out : grid_data_t;
    signal wv_out : grid_valid_t;

    -- Activation flowing out of PE(r,c) to the bottom
    signal a_out : grid_data_t;
    signal av_out : grid_valid_t;

    -- PE accumulator readout
    signal pe_psum : grid_accum_t;

    ---------------------------------------------------------------------------
    -- FSM
    ---------------------------------------------------------------------------
    type state_t is (IDLE, ACCUM, DRAIN, READOUT);
    signal state : state_t;

    -- Drain latency: last input + skew + propagation + 1 MAC pipeline stage.
    -- Worst-case skew = ROWS-1 + COLS-1, propagation = ROWS-1 + COLS-1,
    -- but propagation overlaps with skew. Net: the last unskewed input reaches
    -- PE(ROWS-1,COLS-1) after (ROWS-1)+(COLS-1) cycles of skew + (ROWS-1)+(COLS-1)
    -- cycles of propagation, then 1 MAC cycle.
    -- So drain = 2*(ROWS+COLS) — conservative upper bound.
    constant DRAIN_CYCLES : positive := 2 * (ROWS + COLS);

    signal drain_cnt : integer range 0 to DRAIN_CYCLES;
    signal rd_addr   : integer range 0 to ROWS * COLS - 1;

begin

    ---------------------------------------------------------------------------
    -- Input skew shift registers  (row-major for weights, column-major for acts)
    ---------------------------------------------------------------------------

    -- Weight skew: each row r has an r-deep shift register
    g_wskew_row : for r in 0 to ROWS - 1 generate
        -- Tap the r-th weight from the flat input
        signal w_raw : data_t;
    begin
        w_raw <= signed(i_weight((r + 1) * DATA_WIDTH - 1 downto r * DATA_WIDTH));

        -- Depth-0 (row 0): no skew, direct passthrough
        g_skew_d0 : if r = 0 generate
            skew_w_data(0)(0)  <= w_raw;
            skew_w_valid(0)(0) <= i_weight_valid;
            w_skewed(0)        <= skew_w_data(0)(0);
            wv_skewed(0)       <= skew_w_valid(0)(0);
        end generate;

        -- Depth > 0: shift register
        g_skew_dn : if r > 0 generate
            p_wskew : process (clk) is
            begin
                if rising_edge(clk) then
                    if rstn = '0' then
                        for d in 0 to r loop
                            skew_w_data(r)(d)  <= (others => '0');
                            skew_w_valid(r)(d) <= '0';
                        end loop;
                    else
                        skew_w_data(r)(0)  <= w_raw;
                        skew_w_valid(r)(0) <= i_weight_valid;
                        for d in 1 to r loop
                            skew_w_data(r)(d)  <= skew_w_data(r)(d - 1);
                            skew_w_valid(r)(d) <= skew_w_valid(r)(d - 1);
                        end loop;
                    end if;
                end if;
            end process p_wskew;
            w_skewed(r)  <= skew_w_data(r)(r);
            wv_skewed(r) <= skew_w_valid(r)(r);
        end generate;
    end generate g_wskew_row;

    -- Activation skew: each column c has a c-deep shift register
    g_askew_col : for c in 0 to COLS - 1 generate
        signal a_raw : data_t;
    begin
        a_raw <= signed(i_act((c + 1) * DATA_WIDTH - 1 downto c * DATA_WIDTH));

        g_skew_d0 : if c = 0 generate
            skew_a_data(0)(0)  <= a_raw;
            skew_a_valid(0)(0) <= i_act_valid;
            a_skewed(0)        <= skew_a_data(0)(0);
            av_skewed(0)       <= skew_a_valid(0)(0);
        end generate;

        g_skew_dn : if c > 0 generate
            p_askew : process (clk) is
            begin
                if rising_edge(clk) then
                    if rstn = '0' then
                        for d in 0 to c loop
                            skew_a_data(c)(d)  <= (others => '0');
                            skew_a_valid(c)(d) <= '0';
                        end loop;
                    else
                        skew_a_data(c)(0)  <= a_raw;
                        skew_a_valid(c)(0) <= i_act_valid;
                        for d in 1 to c loop
                            skew_a_data(c)(d)  <= skew_a_data(c)(d - 1);
                            skew_a_valid(c)(d) <= skew_a_valid(c)(d - 1);
                        end loop;
                    end if;
                end if;
            end process p_askew;
            a_skewed(c)  <= skew_a_data(c)(c);
            av_skewed(c) <= skew_a_valid(c)(c);
        end generate;
    end generate g_askew_col;

    ---------------------------------------------------------------------------
    -- PE grid wiring  (structural interconnects)
    --
    -- w_in(r,c)  = left-edge source for PE(r,c)
    -- a_in(r,c)  = top-edge source for PE(r,c)
    --
    -- Left column (c=0): fed by skewed weight for that row
    -- Other columns (c>0): fed by PE(r,c-1) weight passthrough
    --
    -- Top row (r=0): fed by skewed activation for that column
    -- Other rows (r>0): fed by PE(r-1,c) activation passthrough
    ---------------------------------------------------------------------------

    g_row : for r in 0 to ROWS - 1 generate
        g_col : for c in 0 to COLS - 1 generate
        begin
            -- Left-edge weight source
            g_w_edge : if c = 0 generate
                w_in(r, 0)  <= w_skewed(r);
                wv_in(r, 0) <= wv_skewed(r);
            end generate;

            -- Internal weight horizontal wiring
            g_w_prop : if c > 0 generate
                w_in(r, c)  <= w_out(r, c - 1);
                wv_in(r, c) <= wv_out(r, c - 1);
            end generate;

            -- Top-edge activation source
            g_a_edge : if r = 0 generate
                a_in(0, c)  <= a_skewed(c);
                av_in(0, c) <= av_skewed(c);
            end generate;

            -- Internal activation vertical wiring
            g_a_prop : if r > 0 generate
                a_in(r, c)  <= a_out(r - 1, c);
                av_in(r, c) <= av_out(r - 1, c);
            end generate;

            -- PE instantiation
            u_pe : entity work.pe_os
                generic map (
                    DATA_WIDTH  => DATA_WIDTH,
                    ACCUM_WIDTH => ACCUM_WIDTH
                )
                port map (
                    clk      => clk,
                    rstn     => rstn,
                    i_weight => w_in(r, c),
                    i_act    => a_in(r, c),
                    i_valid  => wv_in(r, c) and av_in(r, c),
                    o_weight => w_out(r, c),
                    o_act    => a_out(r, c),
                    o_valid  => wv_out(r, c),
                    i_clear  => '0',
                    o_psum   => pe_psum(r, c)
                );
            -- Note: wv_out propagates weight-side valid downstream.
            -- Both wv_in and av_in must be '1' for a MAC, but data alignment
            -- is guaranteed by the input skew registers, so wv_out suffices
            -- as the row-valid propagation vector.
        end generate g_col;
    end generate g_row;

    ---------------------------------------------------------------------------
    -- FSM: IDLE -> ACCUM -> DRAIN -> READOUT -> IDLE
    ---------------------------------------------------------------------------

    p_fsm : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state     <= IDLE;
                drain_cnt <= 0;
                rd_addr   <= 0;
                o_data    <= (others => '0');
                o_valid   <= '0';
                o_last    <= '0';
                o_channel <= 0;
            else
                -- Registered defaults
                o_valid <= '0';
                o_last  <= '0';

                case state is

                    when IDLE =>
                        rd_addr   <= 0;
                        drain_cnt <= 0;
                        if i_weight_valid = '1' or i_act_valid = '1' then
                            state <= ACCUM;
                        end if;

                    when ACCUM =>
                        -- When the last inner-dimension beat arrives, start draining.
                        if i_weight_last = '1' and i_weight_valid = '1' then
                            state     <= DRAIN;
                            drain_cnt <= 0;
                        end if;

                    when DRAIN =>
                        -- Wait for the pipeline to flush completely.
                        if drain_cnt >= DRAIN_CYCLES - 1 then
                            state   <= READOUT;
                            rd_addr <= 0;
                        else
                            drain_cnt <= drain_cnt + 1;
                        end if;

                    when READOUT =>
                        -- Stream each PE's accumulator sequentially by channel.
                        o_channel <= rd_addr;
                        o_data    <= std_logic_vector(
                            pe_psum(rd_addr / COLS, rd_addr mod COLS)(DATA_WIDTH - 1 downto 0)
                        );
                        o_valid <= '1';

                        if rd_addr = ROWS * COLS - 1 then
                            o_last <= '1';
                            state  <= IDLE;
                        else
                            rd_addr <= rd_addr + 1;
                        end if;

                end case;
            end if;
        end if;
    end process p_fsm;

end architecture rtl;
