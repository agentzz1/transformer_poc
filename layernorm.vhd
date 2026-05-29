-- =============================================================================
-- Fixed-point Layer Normalization for the structural transformer path
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.clog2_pkg.all;

entity layernorm is
    generic (
        DATA_WIDTH : positive := 16;
        VEC_SIZE   : positive := 512
    );
    port (
        clk              : in  std_logic;
        rstn             : in  std_logic;

        i_data           : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid          : in  std_logic;
        i_last           : in  std_logic;
        i_channel        : in  integer;

        o_data           : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid          : out std_logic;
        o_last           : out std_logic;
        o_channel        : out integer;

        i_params_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_params_valid   : in  std_logic;
        i_params_addr    : in  std_logic_vector(clog2(VEC_SIZE) - 1 downto 0);
        i_params_sel     : in  std_logic
    );
end entity layernorm;

architecture rtl of layernorm is

    -- =========================================================================
    -- LayerNorm with LOD-shift 1/sqrt approximation (FleXNNgine-style)
    --   var = sum_sq/N - mean^2
    --   lod = leading_one(var)          ~ 2 * log2(std)
    --   shift = (lod + 1) / 2            ~ log2(std)
    --   y_i = (x_i - mean) >> shift      ~ (x_i - mean) / std
    -- No multiplier in the normalize path, no isqrt iteration.
    -- VEC_SIZE must be a power of two for the >>VEC_BITS mean/var divides.
    -- =========================================================================
    constant VEC_BITS : positive := clog2(VEC_SIZE);
    constant I16_MAX  : integer := 2 ** (DATA_WIDTH - 1) - 1;
    constant I16_MIN  : integer := -(2 ** (DATA_WIDTH - 1));

    -- LayerNorm output headroom: scale standardized output down by 2^HEADROOM
    -- so std~1 values (up to +-3) fit in Q1.7 instead of clipping at +-1.0.
    -- Must match golden_model.LN_HEADROOM and qat_hw_exact.LN_HEADROOM.
    constant LN_HEADROOM : integer := 2;
    constant LN_FRAC     : integer := (DATA_WIDTH - 1) - LN_HEADROOM;

    -- Accumulator widths sized for worst-case VEC_SIZE up to ~1024 with
    -- DATA_WIDTH up to 16:
    --   sum   = N * 2^(DATA_WIDTH-1)               -> fits comfortably in 32-bit
    --   sum_sq= N * 2^(2*(DATA_WIDTH-1))           -> fits comfortably in 48-bit
    constant SUM_W    : positive := 32;
    constant SUMSQ_W  : positive := 48;

    type state_t is (ST_IDLE, ST_ACCUMULATE, ST_COMPUTE, ST_NORMALIZE, ST_DONE);
    signal state : state_t := ST_IDLE;

    type data_array_t is array (0 to VEC_SIZE - 1) of signed(DATA_WIDTH - 1 downto 0);
    signal data_buf : data_array_t := (others => (others => '0'));

    signal sum_reg    : signed(SUM_W - 1 downto 0)   := (others => '0');
    signal sum_sq_reg : unsigned(SUMSQ_W - 1 downto 0) := (others => '0');
    signal mean_reg   : signed(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal var_reg    : unsigned(SUMSQ_W - 1 downto 0) := (others => '0');
    signal norm_shift : integer range 0 to SUMSQ_W := 0;

    signal acc_cnt    : integer range 0 to VEC_SIZE := 0;
    signal norm_cnt   : integer range 0 to VEC_SIZE := 0;
    signal comp_step  : integer range 0 to 2 := 0;
    signal saved_chan : integer := 0;

    -- Force LUT mapping for the squaring multiplier so it doesn't burn a DSP.
    -- (Single 2W x 2W squaring per accumulate cycle; ~64 LUTs.)
    signal sq_v : unsigned(2 * DATA_WIDTH - 1 downto 0) := (others => '0');
    attribute use_dsp : string;
    attribute use_dsp of sq_v : signal is "no";

    -- ---------------------------------------------------------------------
    -- Leading-one detect on the variance accumulator.
    -- Returns the index of the most-significant '1' (0 if value is zero).
    -- ---------------------------------------------------------------------
    function leading_one (x : unsigned) return integer is
    begin
        for i in x'high downto x'low loop
            if x(i) = '1' then
                return i;
            end if;
        end loop;
        return 0;
    end function;

    function sat_out (value : signed) return signed is
    begin
        if value > to_signed(I16_MAX, value'length) then
            return to_signed(I16_MAX, DATA_WIDTH);
        elsif value < to_signed(I16_MIN, value'length) then
            return to_signed(I16_MIN, DATA_WIDTH);
        end if;
        return resize(value, DATA_WIDTH);
    end function;

begin

    p_main : process(clk) is
        variable x_val    : signed(DATA_WIDTH - 1 downto 0);
        variable square_v : unsigned(2 * DATA_WIDTH - 1 downto 0);
        variable mean_sq  : unsigned(SUMSQ_W - 1 downto 0);
        variable mean_sq_full : signed(2 * DATA_WIDTH - 1 downto 0);
        variable sum_sq_div  : unsigned(SUMSQ_W - 1 downto 0);
        -- Widen to 2*DATA_WIDTH so the << (DATA_WIDTH-1) upscale fits without
        -- overflow before the final shift-right and saturation to DATA_WIDTH.
        variable diff_v   : signed(2 * DATA_WIDTH - 1 downto 0);
        variable norm_v   : signed(2 * DATA_WIDTH - 1 downto 0);
        variable lod_i    : integer range 0 to SUMSQ_W - 1;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state      <= ST_IDLE;
                sum_reg    <= (others => '0');
                sum_sq_reg <= (others => '0');
                mean_reg   <= (others => '0');
                var_reg    <= (others => '0');
                norm_shift <= 0;
                acc_cnt    <= 0;
                norm_cnt   <= 0;
                comp_step  <= 0;
                saved_chan <= 0;
                sq_v       <= (others => '0');
                o_data     <= (others => '0');
                o_valid    <= '0';
                o_last     <= '0';
                o_channel  <= 0;
            else
                o_valid <= '0';
                o_last  <= '0';

                case state is

                    when ST_IDLE =>
                        acc_cnt   <= 0;
                        norm_cnt  <= 0;
                        comp_step <= 0;

                        if i_valid = '1' then
                            x_val    := signed(i_data);
                            -- Square in LUTs (use_dsp="no" attribute)
                            square_v := unsigned(x_val * x_val);
                            sq_v     <= square_v;

                            data_buf(0) <= x_val;
                            sum_reg     <= resize(x_val, SUM_W);
                            sum_sq_reg  <= resize(square_v, SUMSQ_W);
                            saved_chan  <= i_channel;

                            if VEC_SIZE = 1 or i_last = '1' then
                                state <= ST_COMPUTE;
                            else
                                acc_cnt <= 1;
                                state   <= ST_ACCUMULATE;
                            end if;
                        end if;

                    when ST_ACCUMULATE =>
                        if i_valid = '1' then
                            x_val    := signed(i_data);
                            square_v := unsigned(x_val * x_val);
                            sq_v     <= square_v;

                            data_buf(acc_cnt) <= x_val;
                            sum_reg           <= sum_reg + resize(x_val, SUM_W);
                            sum_sq_reg        <= sum_sq_reg + resize(square_v, SUMSQ_W);

                            if acc_cnt = VEC_SIZE - 1 or i_last = '1' then
                                comp_step <= 0;
                                state     <= ST_COMPUTE;
                            else
                                acc_cnt <= acc_cnt + 1;
                            end if;
                        end if;

                    when ST_COMPUTE =>
                        case comp_step is
                            when 0 =>
                                -- Mean = sum >> log2(N)
                                mean_reg  <= resize(shift_right(sum_reg, VEC_BITS), DATA_WIDTH);
                                comp_step <= 1;

                            when 1 =>
                                -- Variance = (sum_sq / N) - mean^2.  Clamp to >= 0.
                                mean_sq_full := mean_reg * mean_reg;
                                mean_sq      := resize(unsigned(mean_sq_full), SUMSQ_W);
                                sum_sq_div   := shift_right(sum_sq_reg, VEC_BITS);
                                if sum_sq_div > mean_sq then
                                    var_reg <= sum_sq_div - mean_sq;
                                else
                                    var_reg <= (others => '0');
                                end if;
                                comp_step <= 2;

                            when others =>
                                -- LOD-shift 1/sqrt approximation
                                --   lod   = MSB index of variance
                                --   shift = (lod + 1) / 2     ~ log2(std)
                                lod_i      := leading_one(var_reg);
                                norm_shift <= (lod_i + 1) / 2;
                                norm_cnt   <= 0;
                                state      <= ST_NORMALIZE;
                        end case;

                    when ST_NORMALIZE =>
                        -- Q1.(DATA_WIDTH-1) output WITH headroom:
                        --   norm = (x - mean) * 2^(FRAC) / 2^norm_shift
                        --   FRAC = (DATA_WIDTH-1) - LN_HEADROOM
                        --
                        -- LN_HEADROOM scales the standardized output down by
                        -- 2^HEADROOM so values (std~1, up to +-3) fit in Q1.7
                        -- instead of being hard-clipped at +-1.0.  This is THE
                        -- fix for the LayerNorm saturation accuracy collapse.
                        -- Must match golden_model.LN_HEADROOM and qat_hw_exact.
                        --
                        -- diff is in 2*DATA_WIDTH bits so the upscale cannot overflow.
                        diff_v := resize(data_buf(norm_cnt), 2 * DATA_WIDTH)
                                  - resize(mean_reg, 2 * DATA_WIDTH);

                        if norm_shift <= LN_FRAC then
                            norm_v := shift_left(diff_v, LN_FRAC - norm_shift);
                        else
                            norm_v := shift_right(diff_v, norm_shift - LN_FRAC);
                        end if;

                        o_data    <= std_logic_vector(sat_out(norm_v));
                        o_valid   <= '1';
                        o_channel <= saved_chan + norm_cnt;

                        if norm_cnt = VEC_SIZE - 1 then
                            o_last   <= '1';
                            norm_cnt <= 0;
                            state    <= ST_DONE;
                        else
                            norm_cnt <= norm_cnt + 1;
                        end if;

                    when ST_DONE =>
                        state <= ST_IDLE;

                end case;
            end if;
        end if;
    end process p_main;

end architecture rtl;
