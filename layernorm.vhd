-- =============================================================================
-- Layer Normalization (Post-LN) -- Synthesizable VHDL Module
-- Compatible with the transformer encoder project
-- =============================================================================
--
-- Generics:
--   DATA_WIDTH : Data path width (default 16)
--   VEC_SIZE   : Number of elements per vector (default 512, must be power of 2)
--
-- Ports:
--   clk, rstn  : Clock and active-low reset
--   i_data, i_valid, i_last, i_channel : Streaming input (channel: integer range)
--   o_data, o_valid, o_last, o_channel : Streaming output (channel: integer range)
--   i_params_data, i_params_valid, i_params_addr, i_params_sel : Parameter loading
--
-- Architecture:
--   FSM: IDLE -> ACCUMULATE -> COMPUTE -> NORMALIZE -> DONE -> IDLE
--   256-entry LUT for reciprocal square root with linear interpolation.
--   Gamma / beta loaded via i_params_* ports during IDLE.
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

    use work.utilities.all;

use work.clog2_pkg.all;

entity layernorm is
    generic (
        DATA_WIDTH : positive := 16;
        VEC_SIZE   : positive := 512   -- must be power of 2
    );
    port (
        clk              : in  std_logic;
        rstn             : in  std_logic;

        -- Streaming input
        i_data           : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid          : in  std_logic;
        i_last           : in  std_logic;
        i_channel        : in  integer range 0 to max_size_x - 1;

        -- Streaming output
        o_data           : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid          : out std_logic;
        o_last           : out std_logic;
        o_channel        : out integer range 0 to max_size_x - 1;

        -- Parameter loading (gamma / beta)
        i_params_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_params_valid   : in  std_logic;
        i_params_addr    : in  std_logic_vector(9 downto 0);  -- 512 gamma + 512 beta
        i_params_sel     : in  std_logic                      -- '0' = gamma, '1' = beta
    );
end entity layernorm;

architecture rtl of layernorm is
    ----------------------------------------------------------------------------
    -- Constants
    ----------------------------------------------------------------------------
    constant VEC_BITS        : positive := clog2(VEC_SIZE);
    constant LUT_DEPTH       : positive := 256;
    constant LUT_WIDTH       : positive := 16;
    constant EPSILON         : positive := 1;

    -- Accumulator widths
    constant SUM_ACC_WIDTH   : positive := DATA_WIDTH + VEC_BITS + 2;
    constant SUMSQ_ACC_WIDTH : positive := 2 * DATA_WIDTH + VEC_BITS + 2;

    -- Internal data types
    subtype data_word is signed(DATA_WIDTH - 1 downto 0);
    type data_array is array (0 to VEC_SIZE - 1) of data_word;

    -- LUT entry type
    type lut_array is array (0 to LUT_DEPTH - 1) of unsigned(LUT_WIDTH - 1 downto 0);

    ----------------------------------------------------------------------------
    -- FSM States
    ----------------------------------------------------------------------------
    type state_t is (IDLE, ACCUMULATE, COMPUTE, NORMALIZE, DONE);

    ----------------------------------------------------------------------------
    -- Function: Initialize reciprocal square root LUT
    ----------------------------------------------------------------------------
    function init_rsqrt_lut return lut_array is
        variable lut : lut_array;
        variable x   : real;
        variable y   : real;
    begin
        for i in 0 to LUT_DEPTH - 1 loop
            -- Normalized input covers [0x8000 .. 0xFFFF], i.e. [32768 .. 65535],
            -- in 256 steps of 128.  Add a small scaled-epsilon for stability.
            x := real(32768 + i * 128) + real(EPSILON) * 0.5;
            y := 1.0 / sqrt(x);
            y := y * real(2 ** 14);                     -- Q2.14
            if y >= real(2 ** LUT_WIDTH - 1) then
                lut(i) := to_unsigned(2 ** LUT_WIDTH - 1, LUT_WIDTH);
            elsif y <= 0.0 then
                lut(i) := to_unsigned(0, LUT_WIDTH);
            else
                lut(i) := to_unsigned(integer(round(y)), LUT_WIDTH);
            end if;
        end loop;
        return lut;
    end function init_rsqrt_lut;

    -- The instantiated LUT
    constant RSQRT_LUT : lut_array := init_rsqrt_lut;

    ----------------------------------------------------------------------------
    -- Function: Approximate reciprocal sqrt with linear interpolation
    ----------------------------------------------------------------------------
    function approx_rsqrt (
        variance : unsigned
    ) return signed is
        variable v_plus   : unsigned(variance'range);
        variable lead_pos : integer range -1 to variance'high;
        variable lshift   : integer range 0 to variance'high;
        variable norm_val : unsigned(15 downto 0);
        variable idx      : integer range 0 to LUT_DEPTH - 1;
        variable idx_next : integer range 0 to LUT_DEPTH - 1;
        variable frac     : unsigned(6 downto 0);
        variable y0       : unsigned(LUT_WIDTH - 1 downto 0);
        variable y1       : unsigned(LUT_WIDTH - 1 downto 0);
        variable interp   : unsigned(LUT_WIDTH + 7 - 1 downto 0);
        variable rshift   : integer range 0 to variance'high;
        variable odd_adj  : unsigned(LUT_WIDTH + 7 - 1 downto 0);
        variable result   : signed(DATA_WIDTH - 1 downto 0);
    begin
        -- Step 1: add epsilon
        v_plus := variance + to_unsigned(EPSILON, variance'length);

        -- Step 2: find leading '1'
        lead_pos := -1;
        for k in v_plus'high downto 0 loop
            if v_plus(k) = '1' and lead_pos = -1 then
                lead_pos := k;
            end if;
        end loop;

        if lead_pos < 0 then
            -- variance + epsilon == 0 (should not occur with epsilon >= 1)
            return to_signed(2 ** (DATA_WIDTH - 1) - 1, DATA_WIDTH);
        end if;

        -- Step 2 cont'd: normalize so MSB sits at bit 15
        if lead_pos < 15 then
            lshift   := 15 - lead_pos;
            norm_val := resize(shift_left(v_plus, lshift), 16);
        else
            lshift   := 0;
            norm_val := resize(shift_right(v_plus, lead_pos - 15), 16);
        end if;

        -- Step 3: extract 8-bit index [14:7] and 7-bit fraction [6:0]
        idx  := to_integer(norm_val(14 downto 7));
        frac := norm_val(6 downto 0);

        if idx < LUT_DEPTH - 1 then
            idx_next := idx + 1;
        else
            idx_next := LUT_DEPTH - 1;
        end if;

        -- Step 4: LUT lookup + linear interpolation
        -- y = LUT(idx) + (LUT(idx+1) - LUT(idx)) * frac / 128
        y0 := RSQRT_LUT(idx);
        y1 := RSQRT_LUT(idx_next);

        interp := y0 * 128;
        if y1 > y0 then
            interp := interp + (y1 - y0) * to_integer(frac);
        elsif y0 > y1 then
            interp := interp - (y0 - y1) * to_integer(frac);
        end if;

        -- Step 5: undo normalisation
        -- rsqrt(actual) = rsqrt(normalized) * 2^(lshift/2)
        -- If lshift is odd, also multiply by 1/sqrt(2) ~ 46341/65536
        rshift := lshift / 2;

        odd_adj := interp;
        if (lshift mod 2) = 1 then
            odd_adj := resize(
                interp * to_unsigned(46341, 17) / 65536,
                LUT_WIDTH + 7
            );
        end if;

        -- Step 6: right-shift by rshift, resize to DATA_WIDTH signed
        if rshift < LUT_WIDTH + 7 then
            result := signed(resize(
                shift_right(odd_adj, rshift),
                DATA_WIDTH
            ));
        else
            result := (others => '0');
        end if;

        return result;
    end function approx_rsqrt;

    ----------------------------------------------------------------------------
    -- Parameter storage
    ----------------------------------------------------------------------------
    type param_array is array (0 to VEC_SIZE - 1) of data_word;
    signal gamma : param_array := (others => (others => '0'));
    signal beta  : param_array := (others => (others => '0'));

    ----------------------------------------------------------------------------
    -- Data buffer (holds input during accumulation)
    ----------------------------------------------------------------------------
    signal data_buf : data_array := (others => (others => '0'));

    ----------------------------------------------------------------------------
    -- FSM signals
    ----------------------------------------------------------------------------
    signal state       : state_t := IDLE;
    signal next_state  : state_t;

    ----------------------------------------------------------------------------
    -- Accumulator signals (phase 1)
    ----------------------------------------------------------------------------
    signal accum_cnt     : unsigned(VEC_BITS - 1 downto 0);
    signal sum_acc       : signed(SUM_ACC_WIDTH - 1 downto 0);
    signal sum_sq_acc    : signed(SUMSQ_ACC_WIDTH - 1 downto 0);
    signal acc_done      : std_logic;
    signal saved_chan    : integer range 0 to max_size_x - 1;

    ----------------------------------------------------------------------------
    -- Compute signals (phase 2) - registered
    ----------------------------------------------------------------------------
    signal mean_reg     : signed(DATA_WIDTH - 1 downto 0);
    signal inv_std_reg  : signed(DATA_WIDTH - 1 downto 0);
    signal compute_done : std_logic;

    ----------------------------------------------------------------------------
    -- Normalize signals (phase 3)
    ----------------------------------------------------------------------------
    signal norm_cnt       : unsigned(VEC_BITS - 1 downto 0);
    signal norm_done      : std_logic;

begin

    ----------------------------------------------------------------------------
    -- FSM: Sequential state register
    ----------------------------------------------------------------------------
    p_fsm_seq : process(clk, rstn)
    begin
        if rstn = '0' then
            state <= IDLE;
        elsif rising_edge(clk) then
            state <= next_state;
        end if;
    end process p_fsm_seq;

    ----------------------------------------------------------------------------
    -- FSM: Next-state logic
    ----------------------------------------------------------------------------
    p_fsm_comb : process(state, i_valid, acc_done, compute_done, norm_done)
    begin
        next_state <= state;  -- default: hold
        case state is
            when IDLE =>
                if i_valid = '1' then
                    next_state <= ACCUMULATE;
                end if;

            when ACCUMULATE =>
                if acc_done = '1' then
                    next_state <= COMPUTE;
                end if;

            when COMPUTE =>
                if compute_done = '1' then
                    next_state <= NORMALIZE;
                end if;

            when NORMALIZE =>
                if norm_done = '1' then
                    next_state <= DONE;
                end if;

            when DONE =>
                next_state <= IDLE;

            when others =>
                next_state <= IDLE;
        end case;
    end process p_fsm_comb;

    ----------------------------------------------------------------------------
    -- Parameter loading (during IDLE, but can also happen at any time)
    ----------------------------------------------------------------------------
    p_params : process (clk, rstn)
        variable addr_int : integer range 0 to 511;
    begin
        if rstn = '0' then
            gamma <= (others => (others => '0'));
            beta  <= (others => (others => '0'));
        elsif rising_edge(clk) then
            if i_params_valid = '1' then
                addr_int := to_integer(unsigned(i_params_addr(8 downto 0)));
                if i_params_sel = '0' then
                    gamma(addr_int) <= signed(i_params_data);
                else
                    beta(addr_int) <= signed(i_params_data);
                end if;
            end if;
        end if;
    end process p_params;

    ----------------------------------------------------------------------------
    -- ACCUMULATE Phase: Accumulate sum and sum of squares
    ----------------------------------------------------------------------------
    p_accumulate : process(clk, rstn)
        variable x_val  : data_word;
        variable sq_val : signed(2 * DATA_WIDTH - 1 downto 0);
    begin
        if rstn = '0' then
            sum_acc    <= (others => '0');
            sum_sq_acc <= (others => '0');
            accum_cnt  <= (others => '0');
            acc_done   <= '0';
            data_buf   <= (others => (others => '0'));
            saved_chan <= 0;
        elsif rising_edge(clk) then
            if state = ACCUMULATE then
                if i_valid = '1' then
                    x_val := signed(i_data);

                    -- Capture channel from first valid element
                    if accum_cnt = 0 then
                        saved_chan <= i_channel;
                    end if;

                    -- Buffer input
                    data_buf(to_integer(accum_cnt)) <= x_val;
                    -- Accumulate sum
                    sum_acc <= sum_acc + resize(x_val, SUM_ACC_WIDTH);
                    -- Accumulate sum of squares
                    sq_val := x_val * x_val;
                    sum_sq_acc <= sum_sq_acc + resize(sq_val, SUMSQ_ACC_WIDTH);

                    -- Check for last element
                    if accum_cnt = to_unsigned(VEC_SIZE - 1, VEC_BITS) then
                        acc_done <= '1';
                    else
                        accum_cnt <= accum_cnt + 1;
                        acc_done  <= '0';
                    end if;
                else
                    acc_done <= '0';
                end if;
            else
                accum_cnt <= (others => '0');
                acc_done  <= '0';
            end if;
        end if;
    end process p_accumulate;

    ----------------------------------------------------------------------------
    -- COMPUTE Phase: mean, variance, inv_std
    ----------------------------------------------------------------------------
    p_compute : process(clk, rstn)
        variable sum_sr        : signed(SUM_ACC_WIDTH - 1 downto 0);
        variable sum_sq_sr     : signed(SUMSQ_ACC_WIDTH - 1 downto 0);
        variable sum_sq_right  : signed(SUMSQ_ACC_WIDTH - 1 downto 0);
        variable mean_val      : signed(DATA_WIDTH - 1 downto 0);
        variable mean_sq_val   : signed(2 * DATA_WIDTH - 1 downto 0);
        variable variance_val  : unsigned(SUMSQ_ACC_WIDTH - 1 downto 0);
        variable inv_std_val   : signed(DATA_WIDTH - 1 downto 0);
    begin
        if rstn = '0' then
            mean_reg     <= (others => '0');
            inv_std_reg  <= (others => '0');
            compute_done <= '0';
        elsif rising_edge(clk) then
            if state = COMPUTE then
                -- Capture sum and sum_sq, compute mean
                sum_sr    := sum_acc;
                sum_sq_sr := sum_sq_acc;

                -- mean = sum >> log2(VEC_SIZE)
                mean_val := resize(
                    shift_right(sum_sr, VEC_BITS), DATA_WIDTH
                );
                mean_reg <= mean_val;

                -- variance = (sum_sq >> log2(VEC_SIZE)) - mean^2
                mean_sq_val := mean_val * mean_val;

                -- variance = sum_sq_right - mean^2 (with underflow guard)
                sum_sq_right := resize(shift_right(sum_sq_sr, VEC_BITS), SUMSQ_ACC_WIDTH);

                if sum_sq_right >= resize(mean_sq_val, SUMSQ_ACC_WIDTH) then
                    variance_val := unsigned(sum_sq_right - resize(mean_sq_val, SUMSQ_ACC_WIDTH));
                else
                    variance_val := (others => '0');
                end if;

                -- Reciprocal sqrt via LUT + interpolation
                inv_std_val := approx_rsqrt(variance_val);
                inv_std_reg <= inv_std_val;

                compute_done <= '1';
            else
                compute_done <= '0';
            end if;
        end if;
    end process p_compute;

    ----------------------------------------------------------------------------
    -- NORMALIZE Phase: Stream out normalized data
    -- y[i] = gamma[i] * ((x[i] - mean) * inv_std) + beta[i]
    ----------------------------------------------------------------------------
    p_normalize : process (clk, rstn)
        variable x_val    : data_word;
        variable diff     : signed(DATA_WIDTH downto 0);
        variable scaled   : signed(2 * DATA_WIDTH downto 0);
        variable weighted : signed(2 * DATA_WIDTH downto 0);
        variable y_val    : data_word;
        variable idx      : integer range 0 to VEC_SIZE - 1;
    begin
        if rstn = '0' then
            norm_cnt  <= (others => '0');
            norm_done <= '0';
            o_data    <= (others => '0');
            o_valid   <= '0';
            o_last    <= '0';
            o_channel <= 0;
        elsif rising_edge(clk) then
            if state = NORMALIZE then
                idx   := to_integer(norm_cnt);
                x_val := data_buf(idx);

                -- diff = x - mean
                diff := resize(x_val, DATA_WIDTH + 1) -
                        resize(mean_reg, DATA_WIDTH + 1);

                -- scaled = diff * inv_std
                scaled := diff * inv_std_reg;

                -- weighted = gamma[idx] * scaled
                weighted := resize(gamma(idx), 2 * DATA_WIDTH + 1) * scaled;

                -- Re-quantise and add beta
                y_val := resize(
                    weighted(2 * DATA_WIDTH - 1 downto DATA_WIDTH - 1),
                    DATA_WIDTH
                );
                y_val := y_val + beta(idx);

                -- Drive outputs
                o_data    <= std_logic_vector(y_val);
                o_valid   <= '1';
                o_channel <= saved_chan;

                -- Completion
                if norm_cnt = to_unsigned(VEC_SIZE - 1, VEC_BITS) then
                    o_last    <= '1';
                    norm_done <= '1';
                else
                    o_last   <= '0';
                    norm_cnt <= norm_cnt + 1;
                end if;

            else
                norm_cnt  <= (others => '0');
                norm_done <= '0';
                o_valid   <= '0';
                o_last    <= '0';
            end if;
        end if;
    end process p_normalize;

end architecture rtl;
