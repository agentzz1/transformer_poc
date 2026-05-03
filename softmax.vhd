--------------------------------------------------------------------------------
-- softmax.vhd -- Numerically Stable Softmax for Transformer Attention
--------------------------------------------------------------------------------
-- Two-pass streaming design:
--   Pass 1  (SCAN):   Stream in, find max, buffer raw values in BRAM.
--   Pass 2a (EXP):    Read raw BRAM, compute exp(val-max) via LUT,
--                      accumulate sum, store exp in second BRAM.
--   Pass 2b (NORM):   Read exp BRAM, multiply by 1/sum, stream out.
--
-- Formula:  softmax(x_i) = exp(x_i - max) / sum_j(exp(x_j - max))
--
-- Compatible with the accel library streaming conventions
-- (valid / last / channel pattern, rstn active-low).
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.math_real.all;

    use work.utilities.all;

entity softmax is
    generic (
        DATA_WIDTH : positive := 16;
        SEQ_LEN    : positive := 64
    );
    port (
        clk  : in  std_logic;
        rstn : in  std_logic;

        i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid   : in  std_logic;
        i_last    : in  std_logic;
        i_channel : in  integer range 0 to max_size_x - 1;

        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer range 0 to max_size_x - 1
    );
end entity softmax;

architecture rtl of softmax is

    ---------------------------------------------------------------------------
    -- ceil-log2 (handles n=1 by returning 1)
    ---------------------------------------------------------------------------
    function clog2 (n : positive) return natural is
        variable r : natural := 1;
        variable x : natural := n - 1;
    begin
        while x > 1 loop
            r := r + 1;
            x := x / 2;
        end loop;
        return r;
    end function;

    constant ADDR_WIDTH : positive := clog2(SEQ_LEN);

    ---------------------------------------------------------------------------
    -- Exp LUT:  covers x in [LUT_X_MIN, 0], uniform step, Q2.16 output
    ---------------------------------------------------------------------------
    constant LUT_DEPTH   : positive := 256;
    constant LUT_X_MIN   : real     := -10.0;
    constant LUT_MAX_IDX : natural  := LUT_DEPTH - 1;
    constant LUT_WIDTH   : positive := 18;
    constant LUT_FRAC    : positive := 16;

    type exp_lut_t is array (0 to LUT_MAX_IDX) of unsigned(LUT_WIDTH - 1 downto 0);

    function init_exp_lut return exp_lut_t is
        variable step  : real;
        variable x_val : real;
        variable res   : exp_lut_t;
    begin
        step := -LUT_X_MIN / real(LUT_MAX_IDX);
        for i in 0 to LUT_MAX_IDX loop
            x_val := LUT_X_MIN + real(i) * step;
            res(i) := to_unsigned(
                integer(exp(x_val) * real(2 ** LUT_FRAC)),
                LUT_WIDTH
            );
        end loop;
        return res;
    end function;

    constant EXP_LUT : exp_lut_t := init_exp_lut;

    ---------------------------------------------------------------------------
    -- Reciprocal LUT:  1 / (bucket * step), Q8.16
    ---------------------------------------------------------------------------
    constant RECIP_LUT_DEPTH : positive := 512;
    constant RECIP_LUT_WIDTH : positive := 24;
    type recip_lut_t is array (0 to RECIP_LUT_DEPTH - 1) of unsigned(RECIP_LUT_WIDTH - 1 downto 0);

    function init_recip_lut return recip_lut_t is
        variable res : recip_lut_t;
    begin
        for i in 0 to RECIP_LUT_DEPTH - 1 loop
            res(i) := to_unsigned(
                integer(1.0 / (real(i + 1) * 0.005) * real(2 ** 16)),
                RECIP_LUT_WIDTH
            );
        end loop;
        return res;
    end function;

    constant RECIP_LUT : recip_lut_t := init_recip_lut;

    ---------------------------------------------------------------------------
    -- Sum accumulator width (SEQ_LEN * 2^LUT_FRAC fits in LUT_WIDTH+ADDR_WIDTH)
    ---------------------------------------------------------------------------
    constant SUM_WIDTH : positive := LUT_WIDTH + ADDR_WIDTH;

    ---------------------------------------------------------------------------
    -- State machine
    ---------------------------------------------------------------------------
    type state_t is (
        ST_IDLE,
        ST_PASS1_SCAN,
        ST_PASS2A_EXP,
        ST_PASS2B_NORM,
        ST_DONE
    );

    signal state    : state_t;
    signal done_int : std_logic;

    ---------------------------------------------------------------------------
    -- BRAM  (registered read, single-cycle write)
    ---------------------------------------------------------------------------
    type bram_data_t is array (0 to SEQ_LEN - 1) of signed(DATA_WIDTH - 1 downto 0);

    signal bram_raw : bram_data_t;
    signal bram_exp : bram_data_t;

    signal bram_rd_addr : unsigned(ADDR_WIDTH - 1 downto 0);
    signal bram_raw_rd  : signed(DATA_WIDTH - 1 downto 0);
    signal bram_exp_rd  : signed(DATA_WIDTH - 1 downto 0);

    signal bram_wr_en   : std_logic;
    signal bram_wr_addr : unsigned(ADDR_WIDTH - 1 downto 0);
    signal bram_wr_data : signed(DATA_WIDTH - 1 downto 0);
    signal bram_wr_exp  : std_logic;  -- '0' = raw BRAM, '1' = exp BRAM

    ---------------------------------------------------------------------------
    -- Pass 1 signals
    ---------------------------------------------------------------------------
    signal p1_addr    : unsigned(ADDR_WIDTH - 1 downto 0);
    signal p1_max     : signed(DATA_WIDTH - 1 downto 0);
    signal p1_max_lat : signed(DATA_WIDTH - 1 downto 0);
    signal p1_chan    : integer range 0 to max_size_x - 1;

    ---------------------------------------------------------------------------
    -- Pass 2a signals
    --
    -- Pipeline:
    --   T   : 1st cycle -- issue bram_rd_addr = 0, set p2a_addr = 1
    --   T+1 : data[0] ready on bram_raw_rd; diff, LUT, accumulate, write
    --          exp[0] to exp BRAM, issue bram_rd_addr = 1, p2a_addr = 2
    --   ...
    --   T+N : data[N-1] processed; p2a_addr = N+1
    --   T+SEQ_LEN : data[SEQ_LEN-1] processed, p2a_done set
    --   T+SEQ_LEN+1 : transition to ST_PASS2B_NORM
    ---------------------------------------------------------------------------
    signal p2a_addr    : unsigned(ADDR_WIDTH - 1 downto 0);
    signal p2a_wr_addr : unsigned(ADDR_WIDTH - 1 downto 0);
    signal p2a_active  : std_logic;
    signal p2a_lut_val : unsigned(LUT_WIDTH - 1 downto 0);
    signal p2a_sum     : unsigned(SUM_WIDTH - 1 downto 0);
    signal p2a_done    : std_logic;

    ---------------------------------------------------------------------------
    -- Pass 2b signals
    --
    -- Pipeline:
    --   T   : compute recip from p2a_sum, issue read addr 0
    --   T+1 : exp[0] ready, latch (s1), issue read addr 1
    --   T+2 : multiply exp[0] * recip, latch (s2)
    --   T+3 : requantize, o_valid=1 (s3)
    --   ... one output per cycle after 3-cycle latency ...
    --   T+SEQ_LEN+2 : last output (element SEQ_LEN-1)
    --   T+SEQ_LEN+3 : flush, return to ST_DONE
    ---------------------------------------------------------------------------
    signal p2b_addr      : unsigned(ADDR_WIDTH - 1 downto 0);
    signal p2b_s1_valid  : std_logic;
    signal p2b_s2_valid  : std_logic;
    signal p2b_s3_valid  : std_logic;
    signal p2b_s1_data   : signed(DATA_WIDTH - 1 downto 0);
    signal p2b_product   : unsigned(DATA_WIDTH + RECIP_LUT_WIDTH - 1 downto 0);
    signal p2b_result    : signed(DATA_WIDTH - 1 downto 0);
    signal p2b_recip     : unsigned(RECIP_LUT_WIDTH - 1 downto 0);
    signal p2b_out_count : unsigned(ADDR_WIDTH - 1 downto 0);

    ---------------------------------------------------------------------------
    -- Helper: find MSB range of sum_val and extract 9-bit index
    ---------------------------------------------------------------------------
    function sum_to_recip_idx (
        sum_val : unsigned(SUM_WIDTH - 1 downto 0)
    ) return unsigned is
        variable result : unsigned(8 downto 0);
        variable found  : std_logic;
    begin
        result := (others => '0');
        found  := '0';
        for i in SUM_WIDTH - 1 downto 8 loop
            if sum_val(i) = '1' and found = '0' then
                result := sum_val(i downto i - 8);
                found  := '1';
            end if;
        end loop;
        if found = '0' then
            result := resize(sum_val(7 downto 0), 9);
        end if;
        return result;
    end function;

    ---------------------------------------------------------------------------
    -- Helper: map signed diff (always <= 0 for valid data) to LUT index [0,255]
    --
    -- LUT step = 10.0 / 255 = 0.0392157 input units per index.
    -- idx = |diff| / step = |diff| * 255 / 10 = |diff| * 51 / 2.
    ---------------------------------------------------------------------------
    function diff_to_lut_idx (
        diff : signed(DATA_WIDTH - 1 downto 0)
    ) return unsigned is
        variable u_mag  : unsigned(DATA_WIDTH - 1 downto 0);
        variable scaled : unsigned(15 downto 0);
    begin
        if diff >= 0 then
            return to_unsigned(0, 8);
        end if;
        u_mag  := unsigned(-diff);
        scaled := shift_right(u_mag * to_unsigned(51, 6), 1);
        if scaled > to_unsigned(LUT_MAX_IDX, 16) then
            return to_unsigned(LUT_MAX_IDX, 8);
        end if;
        return scaled(7 downto 0);
    end function;

begin

    ---------------------------------------------------------------------------
    -- Output assignments
    ---------------------------------------------------------------------------
    o_data    <= std_logic_vector(p2b_result);
    o_valid   <= p2b_s3_valid;
    o_last    <= '1' when (p2b_s3_valid = '1' and p2b_out_count = SEQ_LEN - 1) else '0';
    o_channel <= p1_chan;
    done_int  <= '1' when state = ST_DONE else '0';

    ---------------------------------------------------------------------------
    -- BRAM registered read port
    ---------------------------------------------------------------------------
    p_bram_rd : process (clk) is
    begin
        if rising_edge(clk) then
            bram_raw_rd <= bram_raw(to_integer(bram_rd_addr));
            bram_exp_rd <= bram_exp(to_integer(bram_rd_addr));
        end if;
    end process p_bram_rd;

    ---------------------------------------------------------------------------
    -- BRAM write port  (bram_wr_exp selects raw or exp BRAM)
    ---------------------------------------------------------------------------
    p_bram_wr : process (clk) is
    begin
        if rising_edge(clk) then
            if bram_wr_en = '1' then
                if bram_wr_exp = '1' then
                    bram_exp(to_integer(bram_wr_addr)) <= bram_wr_data;
                else
                    bram_raw(to_integer(bram_wr_addr)) <= bram_wr_data;
                end if;
            end if;
        end if;
    end process p_bram_wr;

    ---------------------------------------------------------------------------
    -- Main control state machine
    ---------------------------------------------------------------------------
    p_ctrl : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state       <= ST_IDLE;
                bram_wr_en  <= '0';
                bram_wr_exp <= '0';
            else
                -- One-cycle defaults
                bram_wr_en  <= '0';
                bram_wr_exp <= '0';

                case state is

                    ---------------------------------------------------------------
                    -- IDLE
                    ---------------------------------------------------------------
                    when ST_IDLE =>
                        p1_addr    <= (others => '0');
                        p1_max     <= (others => '0');
                        p1_max_lat <= (others => '0');
                        p1_chan    <= 0;

                        if i_valid = '1' then
                            state   <= ST_PASS1_SCAN;
                            p1_chan <= i_channel;
                            p1_max  <= signed(i_data);
                            -- Buffer first element
                            bram_wr_en   <= '1';
                            bram_wr_addr <= (others => '0');
                            bram_wr_data <= signed(i_data);
                            bram_wr_exp  <= '0';
                            p1_addr      <= to_unsigned(1, ADDR_WIDTH);

                            if i_last = '1' then
                                p1_max_lat <= signed(i_data);
                                state      <= ST_PASS2A_EXP;
                            end if;
                        end if;

                    ---------------------------------------------------------------
                    -- PASS 1: scan input, find max, buffer in bram_raw
                    ---------------------------------------------------------------
                    when ST_PASS1_SCAN =>
                        if i_valid = '1' then
                            bram_wr_en   <= '1';
                            bram_wr_addr <= p1_addr;
                            bram_wr_data <= signed(i_data);
                            bram_wr_exp  <= '0';

                            if signed(i_data) > p1_max then
                                p1_max <= signed(i_data);
                            end if;

                            if i_last = '1' then
                                p1_max_lat <= p1_max;
                                state      <= ST_PASS2A_EXP;
                            else
                                p1_addr <= p1_addr + 1;
                            end if;
                        end if;

                    ---------------------------------------------------------------
                    -- PASS 2a: read raw BRAM -> exp LUT -> accumulate -> store exp
                    ---------------------------------------------------------------
                    when ST_PASS2A_EXP =>
                        bram_rd_addr <= p2a_addr;

                        if p2a_done = '1' then
                            -- All SEQ_LEN elements done; advance to normalization
                            state         <= ST_PASS2B_NORM;
                            p2a_done      <= '0';
                            p2a_active    <= '0';
                            p2b_addr      <= (others => '0');
                            p2b_out_count <= (others => '0');
                            p2b_s1_valid  <= '0';
                            p2b_s2_valid  <= '0';
                            p2b_s3_valid  <= '0';

                        elsif p2a_active = '0' then
                            -- First cycle: issue address 0, prepare counters
                            p2a_addr    <= to_unsigned(1, ADDR_WIDTH);
                            p2a_wr_addr <= (others => '0');
                            p2a_sum     <= (others => '0');
                            p2a_active  <= '1';
                            p2a_done    <= '0';

                        else
                            -- bram_raw_rd holds data from previous cycle's address
                            p2a_lut_val <= EXP_LUT(to_integer(
                                diff_to_lut_idx(bram_raw_rd - p1_max_lat)
                            ));
                            p2a_sum <= p2a_sum + p2a_lut_val;

                            -- Write exp to exp BRAM (truncate Q2.16 to DATA_WIDTH integer)
                            bram_wr_en   <= '1';
                            bram_wr_addr <= p2a_wr_addr;
                            bram_wr_data <= signed(resize(
                                p2a_lut_val(LUT_WIDTH - 1 downto LUT_FRAC),
                                DATA_WIDTH
                            ));
                            bram_wr_exp <= '1';

                            -- Advance
                            p2a_wr_addr <= p2a_wr_addr + 1;
                            p2a_addr    <= p2a_addr + 1;

                            if p2a_wr_addr = SEQ_LEN - 1 then
                                p2a_done   <= '1';
                                p2a_active <= '0';
                            end if;
                        end if;

                    ---------------------------------------------------------------
                    -- PASS 2b: read exp BRAM -> multiply by 1/sum -> output
                    --
                    -- p2b_addr doubles as the read-address counter for the
                    -- first SEQ_LEN cycles and then as a flush counter.
                    -- Read addresses are only issued when p2b_addr < SEQ_LEN;
                    -- after that bram_rd_addr is frozen to avoid out-of-range.
                    --
                    -- Cycle-by-cycle (SEQ_LEN=64):
                    --   C0: recip calc, rd_addr=0, addr<=1
                    --   C1: exp[0] latched (s1), rd_addr=1, addr<=2
                    --   C2: exp[0]*recip (s2), exp[1] latched (s1), rd=2, a<=3
                    --   C3: out exp[0] (s3), exp[1]*recip (s2), rd=3, a<=4
                    --   ...
                    --   C63: exp[61] out,   rd_addr=63, addr<=64
                    --   C64: exp[62] out,   rd frozen (addr=64, no more reads)
                    --   C65: exp[63] out (LAST), addr<=66
                    --   C66: flush, addr=67 -> DONE
                    ---------------------------------------------------------------
                    when ST_PASS2B_NORM =>
                        -- Issue read address only while within BRAM range
                        if p2b_addr < SEQ_LEN then
                            bram_rd_addr <= p2b_addr;
                        else
                            bram_rd_addr <= (others => '0');
                        end if;

                        if p2b_addr = 0 and p2b_s1_valid = '0' then
                            -- Phase 0: compute reciprocal from accumulated sum
                            p2b_recip     <= RECIP_LUT(to_integer(
                                sum_to_recip_idx(p2a_sum)
                            ));
                            p2b_addr      <= to_unsigned(1, ADDR_WIDTH);
                            p2b_s1_valid  <= '0';
                            p2b_s2_valid  <= '0';
                            p2b_s3_valid  <= '0';
                            p2b_out_count <= (others => '0');

                        elsif p2b_addr = SEQ_LEN + 3 then
                            -- Pipeline flushed; return to DONE
                            state         <= ST_DONE;
                            p2b_s1_valid  <= '0';
                            p2b_s2_valid  <= '0';
                            p2b_s3_valid  <= '0';

                        else
                            p2b_addr <= p2b_addr + 1;

                            -- Stage 1: latch BRAM read (only for valid addresses,
                            -- i.e., p2b_addr in [1 .. SEQ_LEN])
                            if p2b_addr > 0 and p2b_addr <= SEQ_LEN then
                                p2b_s1_valid <= '1';
                            else
                                p2b_s1_valid <= '0';
                            end if;
                            p2b_s1_data  <= bram_exp_rd;

                            -- Stage 2: multiply exp * recip
                            p2b_s2_valid <= p2b_s1_valid;
                            if p2b_s1_valid = '1' then
                                p2b_product <= unsigned(abs(p2b_s1_data)) * p2b_recip;
                            end if;

                            -- Stage 3: requantize, drive o_data / o_valid
                            p2b_s3_valid <= p2b_s2_valid;
                            if p2b_s2_valid = '1' then
                                -- Product: (DATA_WIDTH+24) bits, 16 fractional
                                -- from recip Q8.16. Shift right 16, fit DATA_WIDTH.
                                p2b_result <= signed(resize(
                                    shift_right(p2b_product, 16),
                                    DATA_WIDTH
                                ));
                            end if;

                            -- Track count for o_last
                            if p2b_s3_valid = '1' then
                                p2b_out_count <= p2b_out_count + 1;
                            end if;
                        end if;

                    ---------------------------------------------------------------
                    -- DONE: return to IDLE
                    ---------------------------------------------------------------
                    when ST_DONE =>
                        state <= ST_IDLE;

                    when others =>
                        state <= ST_IDLE;

                end case;
            end if;
        end if;
    end process p_ctrl;

end architecture rtl;
