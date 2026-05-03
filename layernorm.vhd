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

    constant VEC_BITS : positive := clog2(VEC_SIZE);
    constant Q_SCALE  : integer := 2 ** (DATA_WIDTH - 1);
    constant I16_MAX  : integer := 2 ** (DATA_WIDTH - 1) - 1;
    constant I16_MIN  : integer := -(2 ** (DATA_WIDTH - 1));

    type state_t is (ST_IDLE, ST_ACCUMULATE, ST_COMPUTE, ST_NORMALIZE, ST_DONE);
    signal state : state_t := ST_IDLE;

    type data_array_t is array (0 to VEC_SIZE - 1) of signed(DATA_WIDTH - 1 downto 0);
    signal data_buf : data_array_t := (others => (others => '0'));

    signal sum_reg     : signed(63 downto 0) := (others => '0');
    signal sum_sq_reg  : signed(63 downto 0) := (others => '0');
    signal mean_reg    : signed(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal var_reg     : signed(63 downto 0) := (others => '0');
    signal inv_std_reg : signed(DATA_WIDTH - 1 downto 0) := (others => '0');

    signal acc_cnt    : integer range 0 to VEC_SIZE := 0;
    signal norm_cnt   : integer range 0 to VEC_SIZE := 0;
    signal comp_step  : integer range 0 to 2 := 0;
    signal saved_chan : integer := 0;

    function sat16 (
        value : signed
    ) return signed is
        variable max_v : signed(value'length - 1 downto 0);
        variable min_v : signed(value'length - 1 downto 0);
    begin
        max_v := to_signed(I16_MAX, value'length);
        min_v := to_signed(I16_MIN, value'length);

        if value > max_v then
            return to_signed(I16_MAX, DATA_WIDTH);
        elsif value < min_v then
            return to_signed(I16_MIN, DATA_WIDTH);
        end if;

        return resize(value, DATA_WIDTH);
    end function;

    function isqrt (
        n : natural
    ) return natural is
        variable x   : natural := n;
        variable res : natural := 0;
        variable bit : natural := 1073741824;
    begin
        while bit > x loop
            bit := bit / 4;
        end loop;

        while bit /= 0 loop
            if x >= res + bit then
                x   := x - (res + bit);
                res := (res / 2) + bit;
            else
                res := res / 2;
            end if;
            bit := bit / 4;
        end loop;

        return res;
    end function;

    function inv_std_q15 (
        var_value : signed
    ) return signed is
        variable var_i  : natural;
        variable root_i : natural;
        variable inv_i  : natural;
    begin
        if var_value <= to_signed(0, var_value'length) then
            return to_signed(I16_MAX, DATA_WIDTH);
        end if;

        var_i := to_integer(var_value);
        if var_i = 0 then
            var_i := 1;
        end if;

        root_i := isqrt(var_i);
        if root_i = 0 then
            root_i := 1;
        end if;

        inv_i := (Q_SCALE * Q_SCALE) / root_i;
        if inv_i > natural(I16_MAX) then
            inv_i := natural(I16_MAX);
        end if;

        return to_signed(integer(inv_i), DATA_WIDTH);
    end function;

begin

    p_main : process(clk) is
        variable x_val     : signed(DATA_WIDTH - 1 downto 0);
        variable square_v  : signed(2 * DATA_WIDTH - 1 downto 0);
        variable mean_ext  : signed(63 downto 0);
        variable var_v     : signed(63 downto 0);
        variable diff_v    : signed(DATA_WIDTH downto 0);
        variable product_v : signed(2 * DATA_WIDTH downto 0);
        variable norm_v    : signed(2 * DATA_WIDTH downto 0);
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state       <= ST_IDLE;
                sum_reg     <= (others => '0');
                sum_sq_reg  <= (others => '0');
                mean_reg    <= (others => '0');
                var_reg     <= (others => '0');
                inv_std_reg <= (others => '0');
                acc_cnt     <= 0;
                norm_cnt    <= 0;
                comp_step   <= 0;
                saved_chan  <= 0;
                o_data      <= (others => '0');
                o_valid     <= '0';
                o_last      <= '0';
                o_channel   <= 0;
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
                            square_v := x_val * x_val;

                            data_buf(0) <= x_val;
                            sum_reg     <= resize(x_val, 64);
                            sum_sq_reg  <= resize(square_v, 64);
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
                            square_v := x_val * x_val;

                            data_buf(acc_cnt) <= x_val;
                            sum_reg           <= sum_reg + resize(x_val, 64);
                            sum_sq_reg        <= sum_sq_reg + resize(square_v, 64);

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
                                mean_reg  <= resize(shift_right(sum_reg, VEC_BITS), DATA_WIDTH);
                                comp_step <= 1;

                            when 1 =>
                                mean_ext := resize(mean_reg * mean_reg, 64);
                                var_v    := shift_right(sum_sq_reg, VEC_BITS) - mean_ext;
                                if var_v < to_signed(0, var_v'length) then
                                    var_reg <= (others => '0');
                                else
                                    var_reg <= var_v;
                                end if;
                                comp_step <= 2;

                            when others =>
                                inv_std_reg <= inv_std_q15(var_reg);
                                norm_cnt    <= 0;
                                state       <= ST_NORMALIZE;
                        end case;

                    when ST_NORMALIZE =>
                        diff_v    := resize(data_buf(norm_cnt), DATA_WIDTH + 1)
                                     - resize(mean_reg, DATA_WIDTH + 1);
                        product_v := diff_v * inv_std_reg;
                        norm_v    := shift_right(product_v, DATA_WIDTH - 1);

                        o_data    <= std_logic_vector(sat16(norm_v));
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
