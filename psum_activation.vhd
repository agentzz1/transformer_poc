--------------------------------------------------------------------------------
-- psum_activation.vhd -- Activation Wrapper for FFN Output Stream
--
-- Drop-in replacement for the original PoC pass-through stub.  Interface is
-- byte-identical; behaviour is now a real GELU using a 256-entry LUT
-- computed at elaboration time via ieee.math_real.
--
-- Format assumption (matches LayerNorm/Softmax convention in this project):
--   Inputs/outputs are signed fixed-point of width DATA_WIDTH, scaled such
--   that 2**(DATA_WIDTH-1) represents the real value 1.0.  In other words
--   values live in approximately [-1.0, +1.0).
--
-- LUT indexing:
--   For DATA_WIDTH = 8, the input is itself an 8-bit signed integer and is
--   used directly to index a 256-entry LUT (bit-pattern reinterpreted as
--   unsigned).
--   For DATA_WIDTH > 8 the upper 8 bits of i_data are used as the LUT index
--   (a coarse quantisation; sufficient for GELU which is smooth) and the
--   LUT output is sign-/precision-extended back to DATA_WIDTH.
--
-- MODE generic:
--   "GELU"     - real GELU activation (default)
--   "IDENTITY" - pass-through (legacy behaviour, useful for debug)
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.math_real.all;

entity psum_activation is
    generic (
        DATA_WIDTH   : integer := 16;
        NUM_ELEMENTS : integer := 2048;
        MODE         : string  := "GELU"
    );
    port (
        clk  : in std_logic;
        rstn : in std_logic;

        start : in std_logic;
        done  : out std_logic;

        i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid   : in  std_logic;
        i_last    : in  std_logic;
        i_channel : in  integer;

        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer);
end entity psum_activation;

architecture rtl of psum_activation is

    -- ── LUT type and elaboration-time initialisation ─────────────────────
    constant LUT_DEPTH : integer := 256;
    type gelu_lut_t is array (0 to LUT_DEPTH - 1) of integer;

    -- GELU(x) approximated with Hendrycks-Gimpel tanh form:
    --   GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    -- LUT entries are int8 values; outer wrapper rescales for DATA_WIDTH /= 8.
    function init_gelu_lut return gelu_lut_t is
        variable res         : gelu_lut_t;
        variable x_int       : integer;
        variable x_real      : real;
        variable y_real      : real;
        variable tanh_arg    : real;
        variable e_pos, e_neg, tanh_val : real;
        variable y_q         : integer;
        constant SQRT_2_PI   : real := 0.7978845608028654;
        constant Q_SCALE_8   : real := 128.0;     -- LUT is always Q1.7 internally
    begin
        for i in 0 to LUT_DEPTH - 1 loop
            -- Interpret index as signed 8-bit
            if i < 128 then
                x_int := i;
            else
                x_int := i - 256;
            end if;
            x_real   := real(x_int) / Q_SCALE_8;
            tanh_arg := SQRT_2_PI * (x_real + 0.044715 * x_real ** 3);
            e_pos    := exp( tanh_arg);
            e_neg    := exp(-tanh_arg);
            tanh_val := (e_pos - e_neg) / (e_pos + e_neg);
            y_real   := 0.5 * x_real * (1.0 + tanh_val);
            -- Truncate toward zero (NOT round): the Python golden model uses
            -- int(y_real*128) which truncates.  VHDL integer() rounds, so we
            -- must trunc() first to stay bit-identical to the reference.
            y_q      := integer(trunc(y_real * Q_SCALE_8));
            if y_q >  127 then y_q :=  127; end if;
            if y_q < -128 then y_q := -128; end if;
            res(i) := y_q;
        end loop;
        return res;
    end function;

    constant GELU_LUT : gelu_lut_t := init_gelu_lut;

    -- ── Pipeline registers (1-cycle latency, same as original stub) ──────
    signal elem_cnt   : integer := 0;
    signal data_reg   : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal valid_reg  : std_logic;
    signal last_reg   : std_logic;
    signal chan_reg   : integer := 0;
    signal done_reg   : std_logic;

    -- Helpers
    function clamp_q (v : integer) return integer is
        variable hi : integer := 2 ** (DATA_WIDTH - 1) - 1;
        variable lo : integer := -(2 ** (DATA_WIDTH - 1));
    begin
        if v > hi then return hi; end if;
        if v < lo then return lo; end if;
        return v;
    end function;

begin

    -- ─────────────────────────────────────────────────────────────────────
    -- Control: track stream length, generate done pulse
    -- ─────────────────────────────────────────────────────────────────────
    proc_ctrl : process(clk, rstn)
    begin
        if rstn = '0' then
            elem_cnt <= 0;
            done_reg <= '0';
        elsif rising_edge(clk) then
            done_reg <= '0';
            if start = '1' then
                elem_cnt <= 0;
            elsif i_valid = '1' then
                if i_last = '1' or elem_cnt = NUM_ELEMENTS - 1 then
                    done_reg <= '1';
                    elem_cnt <= 0;
                else
                    elem_cnt <= elem_cnt + 1;
                end if;
            end if;
        end if;
    end process proc_ctrl;

    done <= done_reg;

    -- ─────────────────────────────────────────────────────────────────────
    -- Datapath: per-cycle LUT lookup + (re)quantisation
    -- ─────────────────────────────────────────────────────────────────────
    proc_pipe : process(clk, rstn)
        variable lut_idx     : integer range 0 to LUT_DEPTH - 1;
        variable in_sig      : signed(DATA_WIDTH - 1 downto 0);
        variable in_top8     : signed(7 downto 0);
        variable y_int8      : integer;
        variable y_scaled    : integer;
    begin
        if rstn = '0' then
            data_reg  <= (others => '0');
            valid_reg <= '0';
            last_reg  <= '0';
            chan_reg  <= 0;
        elsif rising_edge(clk) then
            if i_valid = '1' then
                if MODE = "IDENTITY" then
                    data_reg <= i_data;
                else  -- "GELU" (default)
                    in_sig := signed(i_data);

                    if DATA_WIDTH = 8 then
                        -- Direct 256-entry lookup
                        lut_idx := to_integer(unsigned(i_data));
                        y_int8  := GELU_LUT(lut_idx);
                        data_reg <= std_logic_vector(to_signed(y_int8, DATA_WIDTH));
                    else
                        -- Quantise input to int8 by taking the top 8 bits
                        in_top8 := resize(shift_right(in_sig, DATA_WIDTH - 8), 8);
                        lut_idx := to_integer(unsigned(std_logic_vector(in_top8)));
                        y_int8  := GELU_LUT(lut_idx);
                        -- Rescale int8 (Q1.7) result back to DATA_WIDTH (Q1.W-1)
                        y_scaled := y_int8 * (2 ** (DATA_WIDTH - 8));
                        data_reg <= std_logic_vector(
                            to_signed(clamp_q(y_scaled), DATA_WIDTH));
                    end if;
                end if;
                valid_reg <= '1';
                last_reg  <= i_last;
                chan_reg  <= i_channel;
            else
                valid_reg <= '0';
                last_reg  <= '0';
            end if;
        end if;
    end process proc_pipe;

    o_data    <= data_reg;
    o_valid   <= valid_reg;
    o_last    <= last_reg;
    o_channel <= chan_reg;

end architecture rtl;
