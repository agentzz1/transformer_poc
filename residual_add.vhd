library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

-- Required for clog2
library work;
use work.clog2_pkg.all;
use work.utilities.all;

entity residual_add is
  generic (
    DATA_WIDTH : positive := 16;
    VEC_SIZE   : positive := 512
  );
  port (
    clk            : in  std_logic;
    rstn           : in  std_logic;

    -- Main data path (e.g., MHA output)
    i_data         : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    i_data_valid   : in  std_logic;
    i_data_last    : in  std_logic;
    i_data_channel : in  integer;

    -- Residual data path (e.g., original input)
    i_residual         : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    i_residual_valid   : in  std_logic;
    i_residual_last    : in  std_logic;
    i_residual_channel : in  integer;

    -- Ready backpressure to the producers
    o_ready            : out std_logic;

    -- Output to the next module
    o_data    : out std_logic_vector(DATA_WIDTH-1 downto 0);
    o_valid   : out std_logic;
    o_last    : out std_logic;
    o_channel : out integer;

    -- LayerNorm parameters (wired to internal LN)
    i_params_data  : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    i_params_valid : in  std_logic;
    i_params_addr  : in  std_logic_vector(clog2(VEC_SIZE)-1 downto 0);
    i_params_sel   : in  std_logic
  );
end entity residual_add;

architecture rtl of residual_add is

  function sat16_int (
    value : integer
  ) return signed is
  begin
    if value > 2 ** (DATA_WIDTH - 1) - 1 then
      return to_signed(2 ** (DATA_WIDTH - 1) - 1, DATA_WIDTH);
    elsif value < -(2 ** (DATA_WIDTH - 1)) then
      return to_signed(-(2 ** (DATA_WIDTH - 1)), DATA_WIDTH);
    end if;

    return to_signed(value, DATA_WIDTH);
  end function;

  type t_fsm_state is (ST_IDLE, ST_ADD_ELEMENTS, ST_LAYERNORM, ST_DONE);
  signal state : t_fsm_state;

  -- Internal LayerNorm signals
  signal ln_data_in    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal ln_valid_in   : std_logic;
  signal ln_last_in    : std_logic;
  signal ln_channel_in : integer := 0;

  signal ln_data_out    : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal ln_valid_out   : std_logic;
  signal ln_last_out    : std_logic;
  signal ln_channel_out : integer := 0;

  component layernorm is
    generic (
      DATA_WIDTH : positive := 16;
      VEC_SIZE   : positive := 512
    );
    port (
      clk            : in  std_logic;
      rstn           : in  std_logic;
      i_data         : in  std_logic_vector(DATA_WIDTH-1 downto 0);
      i_valid        : in  std_logic;
      i_last         : in  std_logic;
      i_channel      : in  integer;
      i_params_data  : in  std_logic_vector(DATA_WIDTH-1 downto 0);
      i_params_valid : in  std_logic;
      i_params_addr  : in  std_logic_vector(clog2(VEC_SIZE)-1 downto 0);
      i_params_sel   : in  std_logic;
      o_data         : out std_logic_vector(DATA_WIDTH-1 downto 0);
      o_valid        : out std_logic;
      o_last         : out std_logic;
      o_channel      : out integer
    );
  end component;

begin

  ----------------------------------------------------------------------------
  -- Combined Synchronous FSM
  ----------------------------------------------------------------------------
  p_fsm : process(clk) is
    variable sum_val : integer;
  begin
    if rising_edge(clk) then
      if rstn = '0' then
        state <= ST_IDLE;
        o_ready <= '1';
        ln_valid_in <= '0';
        ln_last_in <= '0';
        ln_data_in <= (others => '0');
        ln_channel_in <= 0;
      else
        -- Default: stop LN input
        ln_valid_in <= '0';
        ln_last_in  <= '0';

        case state is
          when ST_IDLE =>
            o_ready <= '1';
            if i_data_valid = '1' or i_residual_valid = '1' then
               report "RES_ADD: IDLE data_v=" & std_logic'image(i_data_valid) & " res_v=" & std_logic'image(i_residual_valid);
            end if;
            if i_data_valid = '1' and i_residual_valid = '1' then
               sum_val := to_integer(signed(i_data)) + to_integer(signed(i_residual));
               ln_data_in    <= std_logic_vector(sat16_int(sum_val));
               ln_valid_in   <= '1';
              ln_last_in    <= i_data_last;
              ln_channel_in <= i_data_channel;
              
              if i_data_last = '1' then
                state <= ST_LAYERNORM;
                o_ready <= '0';
              else
                state <= ST_ADD_ELEMENTS;
              end if;
            end if;

          when ST_ADD_ELEMENTS =>
            o_ready <= '1';
            if i_data_valid = '1' and i_residual_valid = '1' then
              sum_val := to_integer(signed(i_data)) + to_integer(signed(i_residual));
              ln_data_in    <= std_logic_vector(sat16_int(sum_val));
              ln_valid_in   <= '1';
              ln_last_in    <= i_data_last;
              ln_channel_in <= i_data_channel;
              
              if i_data_last = '1' then
                state <= ST_LAYERNORM;
                o_ready <= '0';
              end if;
            end if;

          when ST_LAYERNORM =>
            o_ready <= '0';
            if ln_valid_out = '1' and ln_last_out = '1' then
              state <= ST_DONE;
            end if;

          when ST_DONE =>
            o_ready <= '0';
            state <= ST_IDLE;

          when others =>
            state <= ST_IDLE;
        end case;
      end if;
    end if;
  end process p_fsm;

  ----------------------------------------------------------------------------
  -- Output Wiring
  ----------------------------------------------------------------------------
  o_data    <= ln_data_out;
  o_valid   <= ln_valid_out;
  o_last    <= ln_last_out;
  o_channel <= ln_channel_out;

  ----------------------------------------------------------------------------
  -- LayerNorm Instance
  ----------------------------------------------------------------------------
  u_layernorm : layernorm
    generic map (
      DATA_WIDTH => DATA_WIDTH,
      VEC_SIZE   => VEC_SIZE
    )
    port map (
      clk            => clk,
      rstn           => rstn,
      i_data         => ln_data_in,
      i_valid        => ln_valid_in,
      i_last         => ln_last_in,
      i_channel      => ln_channel_in,
      i_params_data  => i_params_data,
      i_params_valid => i_params_valid,
      i_params_addr  => i_params_addr,
      i_params_sel   => i_params_sel,
      o_data         => ln_data_out,
      o_valid        => ln_valid_out,
      o_last         => ln_last_out,
      o_channel      => ln_channel_out
    );

end architecture rtl;
