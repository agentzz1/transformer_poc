-- ============================================================================
-- scalar_ops.vhd  --  Scalar Operations for Transformer Activation & Norm
-- ============================================================================
-- Provides two independent, fully-pipelined scalar operations used within
-- Transformer encoder/decoder blocks:
--
--   1) GELU activation  (Gaussian Error Linear Unit)
--        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
--        Implemented via a direct lookup table with linear interpolation
--        between adjacent sample points.  The tanh component is implicitly
--        captured in the pre-computed table.
--
--   2) Reciprocal square-root  (1/sqrt(x))
--        Used by LayerNorm to normalize by 1/sqrt(variance + epsilon).
--        Implemented via a lookup table with linear interpolation.
--        Input zero is clamped to a configurable minimum to avoid division
--        by zero.
--
-- Both datapaths are independent and may be used simultaneously.
-- Pipeline latency = PIPE_LEN cycles (minimum = 3).
--
-- Fixed-point format: Q4.12 (sign + 3 integer + 12 fractional bits)
--   Range:  [-8.0, +7.999756],  resolution = 1/4096
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.utilities.all;

entity scalar_ops is
    generic (
        DATA_WIDTH : positive := 16;
        PIPE_LEN   : positive := 4;
        LUT_SIZE   : positive := 256
    );
    port (
        clk  : in std_logic;
        rstn : in std_logic;

        -- GELU activation interface
        i_gelu_data  : in  signed(DATA_WIDTH - 1 downto 0);
        i_gelu_valid : in  std_logic;
        o_gelu_data  : out signed(DATA_WIDTH - 1 downto 0);
        o_gelu_valid : out std_logic;

        -- Reciprocal square-root interface (LayerNorm)
        i_div_data  : in  unsigned(DATA_WIDTH - 1 downto 0);
        i_div_valid : in  std_logic;
        o_div_result : out unsigned(DATA_WIDTH - 1 downto 0);
        o_div_valid  : out std_logic
    );
end entity scalar_ops;

architecture rtl of scalar_ops is

    constant ADDR_WIDTH : positive := clog2(LUT_SIZE);
    constant FRAC_WIDTH : positive := DATA_WIDTH - ADDR_WIDTH;
    constant LUT_ENTRIES : positive := LUT_SIZE + 1;

    constant PAD_DEPTH : natural := 0 when PIPE_LEN <= 3
                                   else PIPE_LEN - 3;

    constant Q_FRAC  : positive := 12;
    constant Q_SCALE : real     := real(2**Q_FRAC);
    constant Q_MAX   : real     := real(2**(DATA_WIDTH-1) - 1) / Q_SCALE;
    constant Q_MIN   : real     := -real(2**(DATA_WIDTH-1)) / Q_SCALE;

    constant RSQRT_EPSILON : real := 1.0e-6;

    type gelu_lut_t  is array (0 to LUT_ENTRIES - 1) of signed(DATA_WIDTH - 1 downto 0);
    type rsqrt_lut_t is array (0 to LUT_ENTRIES - 1) of unsigned(DATA_WIDTH - 1 downto 0);

    function init_gelu_lut return gelu_lut_t is
        variable lut   : gelu_lut_t;
        variable step  : real;
        variable x_val : real;
        variable arg   : real;
        variable gelu  : real;
        variable qv    : integer;
    begin
        step := (Q_MAX - Q_MIN) / real(LUT_SIZE);
        for i in 0 to LUT_ENTRIES - 1 loop
            x_val := Q_MIN + real(i) * step;
            arg   := SQRT(2.0 / MATH_PI)
                     * (x_val + 0.044715 * x_val * x_val * x_val);
            gelu  := 0.5 * x_val * (1.0 + TANH(arg));
            if gelu > Q_MAX then gelu := Q_MAX; end if;
            if gelu < Q_MIN then gelu := Q_MIN; end if;
            qv    := integer(ROUND(gelu * Q_SCALE));
            lut(i) := to_signed(qv, DATA_WIDTH);
        end loop;
        return lut;
    end function init_gelu_lut;

    function init_rsqrt_lut return rsqrt_lut_t is
        variable lut    : rsqrt_lut_t;
        variable max_in : real;
        variable step   : real;
        variable x_val  : real;
        variable rsqrt  : real;
        variable qv     : integer;
    begin
        max_in := real(2**DATA_WIDTH - 1);
        step   := max_in / real(LUT_SIZE);
        for i in 0 to LUT_ENTRIES - 1 loop
            x_val := real(i) * step;
            if x_val <= RSQRT_EPSILON then
                qv := 2**DATA_WIDTH - 1;
            else
                rsqrt := 1.0 / SQRT(x_val);
                if rsqrt > max_in then rsqrt := max_in; end if;
                qv := integer(ROUND(rsqrt));
            end if;
            lut(i) := to_unsigned(qv, DATA_WIDTH);
        end loop;
        return lut;
    end function init_rsqrt_lut;

    constant GELU_LUT  : gelu_lut_t  := init_gelu_lut;
    constant RSQRT_LUT : rsqrt_lut_t := init_rsqrt_lut;

    -- GELU pipeline
    signal gelu_valid_sr : std_logic_vector(PIPE_LEN - 1 downto 0);
    signal gelu_addr     : unsigned(ADDR_WIDTH - 1 downto 0);
    signal gelu_frac     : unsigned(FRAC_WIDTH - 1 downto 0);
    signal gelu_a1       : unsigned(ADDR_WIDTH - 1 downto 0);
    signal gelu_f1       : unsigned(FRAC_WIDTH - 1 downto 0);
    signal gelu_y0       : signed(DATA_WIDTH - 1 downto 0);
    signal gelu_y1       : signed(DATA_WIDTH - 1 downto 0);
    signal gelu_result   : signed(DATA_WIDTH - 1 downto 0);
    type gelu_pad_vec_t is array (natural range <>) of signed(DATA_WIDTH - 1 downto 0);
    signal gelu_pad_sr   : gelu_pad_vec_t(0 to PAD_DEPTH);

    -- RSQRT pipeline
    signal rsqrt_valid_sr : std_logic_vector(PIPE_LEN - 1 downto 0);
    signal rsqrt_addr     : unsigned(ADDR_WIDTH - 1 downto 0);
    signal rsqrt_frac     : unsigned(FRAC_WIDTH - 1 downto 0);
    signal rsqrt_a1       : unsigned(ADDR_WIDTH - 1 downto 0);
    signal rsqrt_f1       : unsigned(FRAC_WIDTH - 1 downto 0);
    signal rsqrt_y0       : unsigned(DATA_WIDTH - 1 downto 0);
    signal rsqrt_y1       : unsigned(DATA_WIDTH - 1 downto 0);
    signal rsqrt_result   : unsigned(DATA_WIDTH - 1 downto 0);
    type rsqrt_pad_vec_t is array (natural range <>) of unsigned(DATA_WIDTH - 1 downto 0);
    signal rsqrt_pad_sr   : rsqrt_pad_vec_t(0 to PAD_DEPTH);

begin

    -- =========================================================================
    --  GELU Activation pipeline
    -- =========================================================================

    p_gelu : process (clk)
        variable v_biased : signed(DATA_WIDTH downto 0);
        variable v_addr1  : unsigned(ADDR_WIDTH - 1 downto 0);
        variable v_diff   : signed(DATA_WIDTH downto 0);
        variable v_prod   : signed(DATA_WIDTH + FRAC_WIDTH downto 0);
        variable v_base   : signed(DATA_WIDTH + FRAC_WIDTH + 1 downto 0);
        variable v_sum    : signed(DATA_WIDTH + FRAC_WIDTH + 1 downto 0);
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                gelu_valid_sr <= (others => '0');
                gelu_addr  <= (others => '0');
                gelu_frac  <= (others => '0');
                gelu_a1    <= (others => '0');
                gelu_f1    <= (others => '0');
                gelu_y0    <= (others => '0');
                gelu_y1    <= (others => '0');
                gelu_result <= (others => '0');
                for k in gelu_pad_sr'range loop
                    gelu_pad_sr(k) <= (others => '0');
                end loop;
            else
                gelu_valid_sr(0) <= i_gelu_valid;
                gelu_valid_sr(gelu_valid_sr'high downto 1)
                    <= gelu_valid_sr(gelu_valid_sr'high - 1 downto 0);

                -- Stage 0: addr / frac from biased input
                v_biased := resize(i_gelu_data, DATA_WIDTH + 1)
                           + to_signed(2**(DATA_WIDTH-1), DATA_WIDTH + 1);
                gelu_addr <= unsigned(std_logic_vector(
                    v_biased(DATA_WIDTH - 1 downto FRAC_WIDTH)));
                gelu_frac <= unsigned(std_logic_vector(
                    v_biased(FRAC_WIDTH - 1 downto 0)));

                -- Stage 1: LUT read
                gelu_a1 <= gelu_addr;
                gelu_f1 <= gelu_frac;
                if to_integer(gelu_addr) >= LUT_ENTRIES - 1 then
                    v_addr1 := to_unsigned(LUT_ENTRIES - 1, ADDR_WIDTH);
                else
                    v_addr1 := gelu_addr + 1;
                end if;
                gelu_y0 <= GELU_LUT(to_integer(gelu_addr));
                gelu_y1 <= GELU_LUT(to_integer(v_addr1));

                -- Stage 2: interpolation with rounding
                v_diff := resize(gelu_y1, DATA_WIDTH + 1)
                        - resize(gelu_y0, DATA_WIDTH + 1);
                v_prod := v_diff * signed('0' & gelu_f1);
                v_base := (others => gelu_y0(gelu_y0'high));
                v_base(DATA_WIDTH + FRAC_WIDTH - 1 downto FRAC_WIDTH) := gelu_y0;
                v_sum  := v_base + resize(v_prod, DATA_WIDTH + FRAC_WIDTH + 1);
                if v_sum(FRAC_WIDTH - 1) = '1' then
                    gelu_result <= resize(
                        v_sum(DATA_WIDTH + FRAC_WIDTH downto FRAC_WIDTH) + 1,
                        DATA_WIDTH);
                else
                    gelu_result <= resize(
                        v_sum(DATA_WIDTH + FRAC_WIDTH downto FRAC_WIDTH),
                        DATA_WIDTH);
                end if;

                -- Padding
                gelu_pad_sr(0) <= gelu_result;
                for p in 1 to PAD_DEPTH loop
                    gelu_pad_sr(p) <= gelu_pad_sr(p - 1);
                end loop;
            end if;
        end if;
    end process p_gelu;

    o_gelu_valid <= gelu_valid_sr(PIPE_LEN - 1);
    o_gelu_data  <= gelu_pad_sr(PAD_DEPTH);

    -- =========================================================================
    --  Reciprocal Square-Root pipeline
    -- =========================================================================

    p_rsqrt : process (clk)
        variable v_addr1 : unsigned(ADDR_WIDTH - 1 downto 0);
        variable v_diff  : signed(DATA_WIDTH downto 0);
        variable v_prod  : signed(DATA_WIDTH + FRAC_WIDTH downto 0);
        variable v_base  : unsigned(DATA_WIDTH + FRAC_WIDTH + 1 downto 0);
        variable v_sum   : signed(DATA_WIDTH + FRAC_WIDTH + 1 downto 0);
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                rsqrt_valid_sr <= (others => '0');
                rsqrt_addr  <= (others => '0');
                rsqrt_frac  <= (others => '0');
                rsqrt_a1    <= (others => '0');
                rsqrt_f1    <= (others => '0');
                rsqrt_y0    <= (others => '0');
                rsqrt_y1    <= (others => '0');
                rsqrt_result <= (others => '0');
                for k in rsqrt_pad_sr'range loop
                    rsqrt_pad_sr(k) <= (others => '0');
                end loop;
            else
                rsqrt_valid_sr(0) <= i_div_valid;
                rsqrt_valid_sr(rsqrt_valid_sr'high downto 1)
                    <= rsqrt_valid_sr(rsqrt_valid_sr'high - 1 downto 0);

                -- Stage 0: addr / frac from unsigned input
                rsqrt_addr <= i_div_data(DATA_WIDTH - 1 downto FRAC_WIDTH);
                rsqrt_frac <= i_div_data(FRAC_WIDTH - 1 downto 0);

                -- Stage 1: LUT read
                rsqrt_a1 <= rsqrt_addr;
                rsqrt_f1 <= rsqrt_frac;
                if to_integer(rsqrt_addr) >= LUT_ENTRIES - 1 then
                    v_addr1 := to_unsigned(LUT_ENTRIES - 1, ADDR_WIDTH);
                else
                    v_addr1 := rsqrt_addr + 1;
                end if;
                rsqrt_y0 <= RSQRT_LUT(to_integer(rsqrt_addr));
                rsqrt_y1 <= RSQRT_LUT(to_integer(v_addr1));

                -- Stage 2: interpolation (signed diff for monotonic decreasing)
                v_diff := signed('0' & rsqrt_y1) - signed('0' & rsqrt_y0);
                v_prod := v_diff * signed('0' & rsqrt_f1);
                v_base := (others => '0');
                v_base(DATA_WIDTH + FRAC_WIDTH - 1 downto FRAC_WIDTH)
                    := resize(rsqrt_y0, DATA_WIDTH);
                v_sum := signed('0' & v_base)
                       + resize(v_prod, DATA_WIDTH + FRAC_WIDTH + 1);
                if v_sum(FRAC_WIDTH - 1) = '1' then
                    rsqrt_result <= unsigned(resize(
                        v_sum(DATA_WIDTH + FRAC_WIDTH downto FRAC_WIDTH) + 1,
                        DATA_WIDTH));
                else
                    rsqrt_result <= unsigned(resize(
                        v_sum(DATA_WIDTH + FRAC_WIDTH downto FRAC_WIDTH),
                        DATA_WIDTH));
                end if;

                -- Padding
                rsqrt_pad_sr(0) <= rsqrt_result;
                for p in 1 to PAD_DEPTH loop
                    rsqrt_pad_sr(p) <= rsqrt_pad_sr(p - 1);
                end loop;
            end if;
        end if;
    end process p_rsqrt;

    o_div_valid  <= rsqrt_valid_sr(PIPE_LEN - 1);
    o_div_result <= rsqrt_pad_sr(PAD_DEPTH);

end architecture rtl;
