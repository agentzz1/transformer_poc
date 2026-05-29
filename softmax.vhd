--------------------------------------------------------------------------------
-- softmax.vhd -- Fixed-point row softmax for the structural transformer path
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.math_real.all;

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
        i_channel : in  integer;

        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer
    );
end entity softmax;

architecture rtl of softmax is

    constant Q_SCALE     : integer := 2 ** (DATA_WIDTH - 1);
    constant I16_MAX     : integer := 2 ** (DATA_WIDTH - 1) - 1;
    constant I16_MIN     : integer := -(2 ** (DATA_WIDTH - 1));
    constant LUT_DEPTH   : positive := 256;
    constant LUT_MAX_IDX : integer := LUT_DEPTH - 1;
    constant LUT_X_MIN   : real := -10.0;

    type state_t is (ST_IDLE, ST_COLLECT, ST_EXP, ST_OUTPUT);
    signal state : state_t := ST_IDLE;

    type data_array_t is array (0 to SEQ_LEN - 1) of signed(DATA_WIDTH - 1 downto 0);
    -- Range-constrained so Vivado sizes exp_buf as (DATA_WIDTH-1)-bit registers,
    -- not 32-bit integers -- critical for keeping the ST_OUTPUT divider small.
    type int_array_t is array (0 to SEQ_LEN - 1) of integer range 0 to I16_MAX;
    type exp_lut_t is array (0 to LUT_DEPTH - 1) of integer;

    function init_exp_lut return exp_lut_t is
        variable res   : exp_lut_t;
        variable step  : real;
        variable x_val : real;
    begin
        step := -LUT_X_MIN / real(LUT_MAX_IDX);
        for i in 0 to LUT_DEPTH - 1 loop
            x_val := LUT_X_MIN + real(i) * step;
            res(i) := integer(exp(x_val) * real(2 ** 16));
        end loop;
        return res;
    end function;

    constant EXP_LUT_Q16 : exp_lut_t := init_exp_lut;

    signal raw_buf     : data_array_t;
    signal exp_buf     : int_array_t := (others => 0);
    signal max_reg     : signed(DATA_WIDTH - 1 downto 0) := (others => '0');
    -- Range-constrained: max = SEQ_LEN * I16_MAX  (e.g. 16*127=2032 → 11 bits)
    signal sum_exp     : integer range 0 to SEQ_LEN * I16_MAX := 0;
    signal in_count    : integer range 0 to SEQ_LEN := 0;
    signal exp_count   : integer range 0 to SEQ_LEN := 0;
    signal out_count   : integer range 0 to SEQ_LEN := 0;
    signal chan_reg    : integer := 0;

    function sat16_int (
        value : integer
    ) return signed is
    begin
        if value > I16_MAX then
            return to_signed(I16_MAX, DATA_WIDTH);
        elsif value < I16_MIN then
            return to_signed(I16_MIN, DATA_WIDTH);
        end if;

        return to_signed(value, DATA_WIDTH);
    end function;

    function exp_q15_from_diff (
        diff : signed
    ) return integer is
        -- Range constraints keep Vivado from expanding these to 32-bit,
        -- drastically shrinking the combinational divider in ST_EXP.
        variable diff_i  : integer;                                -- full range for compare
        variable mag_val : integer;
        variable mag_i   : integer range 0 to Q_SCALE;            -- capped to Q_SCALE
        variable scaled  : integer range 0 to LUT_MAX_IDX;        -- 8-bit (0..255)
        variable idx     : integer range 0 to LUT_MAX_IDX;        -- 8-bit LUT address
        variable exp_val : integer;
        variable exp_q15 : integer range 0 to I16_MAX;            -- output width
    begin
        diff_i := to_integer(diff);
        if diff_i >= 0 then
            return I16_MAX;
        end if;

        mag_val := -diff_i;
        if mag_val > Q_SCALE then
            mag_val := Q_SCALE;
        end if;
        mag_i := mag_val;

        scaled := (mag_i * LUT_MAX_IDX) / (10 * Q_SCALE);
        if scaled > LUT_MAX_IDX then
            scaled := LUT_MAX_IDX;
        end if;

        idx := LUT_MAX_IDX - scaled;
        exp_val := EXP_LUT_Q16(idx) / 512;   -- >>9 converts Q16 to Q7 (matches Python)
        if exp_val > I16_MAX then
            exp_val := I16_MAX;
        end if;
        exp_q15 := exp_val;

        return exp_q15;
    end function;

begin

    p_ctrl : process (clk) is
        variable diff_v : signed(DATA_WIDTH downto 0);
        variable exp_v  : integer range 0 to I16_MAX;
        -- prod_v width = (DATA_WIDTH-1) + log2(Q_SCALE) = 14 bits for DW=8
        variable prod_v : integer range 0 to I16_MAX * Q_SCALE;
        -- prob_v saturates to I16_MAX; range tells Vivado to size output narrowly
        variable prob_v : integer range 0 to I16_MAX * Q_SCALE;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state     <= ST_IDLE;
                max_reg   <= (others => '0');
                sum_exp   <= 0;
                in_count  <= 0;
                exp_count <= 0;
                out_count <= 0;
                chan_reg  <= 0;
                o_data    <= (others => '0');
                o_valid   <= '0';
                o_last    <= '0';
                o_channel <= 0;
            else
                o_valid <= '0';
                o_last  <= '0';

                case state is

                    when ST_IDLE =>
                        in_count  <= 0;
                        exp_count <= 0;
                        out_count <= 0;
                        sum_exp   <= 0;

                        if i_valid = '1' then
                            raw_buf(0) <= signed(i_data);
                            max_reg    <= signed(i_data);
                            chan_reg   <= i_channel;

                            if i_last = '1' or SEQ_LEN = 1 then
                                state     <= ST_EXP;
                                exp_count <= 0;
                            else
                                in_count <= 1;
                                state    <= ST_COLLECT;
                            end if;
                        end if;

                    when ST_COLLECT =>
                        if i_valid = '1' then
                            raw_buf(in_count) <= signed(i_data);
                            if signed(i_data) > max_reg then
                                max_reg <= signed(i_data);
                            end if;

                            if i_last = '1' or in_count = SEQ_LEN - 1 then
                                exp_count <= 0;
                                sum_exp   <= 0;
                                state     <= ST_EXP;
                            else
                                in_count <= in_count + 1;
                            end if;
                        end if;

                    when ST_EXP =>
                        diff_v := resize(raw_buf(exp_count), DATA_WIDTH + 1)
                                  - resize(max_reg, DATA_WIDTH + 1);
                        exp_v := exp_q15_from_diff(diff_v);
                        exp_buf(exp_count) <= exp_v;
                        sum_exp <= sum_exp + exp_v;

                        if exp_count = SEQ_LEN - 1 then
                            out_count <= 0;
                            state     <= ST_OUTPUT;
                        else
                            exp_count <= exp_count + 1;
                        end if;

                    when ST_OUTPUT =>
                        if sum_exp <= 0 then
                            prob_v := Q_SCALE / SEQ_LEN;
                        else
                            -- Use explicit intermediate so Vivado sees a narrow
                            -- division: prod_v is (DATA_WIDTH-1)+log2(Q_SCALE) bits,
                            -- sum_exp is log2(SEQ_LEN*I16_MAX) bits.  Much smaller
                            -- carry chain than the default 32-bit integer divider.
                            prod_v := exp_buf(out_count) * Q_SCALE;
                            prob_v := prod_v / sum_exp;
                        end if;

                        o_data    <= std_logic_vector(sat16_int(prob_v));
                        o_valid   <= '1';
                        o_channel <= chan_reg;

                        if out_count = SEQ_LEN - 1 then
                            o_last    <= '1';
                            out_count <= 0;
                            state     <= ST_IDLE;
                        else
                            out_count <= out_count + 1;
                        end if;

                end case;
            end if;
        end if;
    end process p_ctrl;

end architecture rtl;
