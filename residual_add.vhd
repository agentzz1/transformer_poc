-------------------------------------------------------------------------------
-- residual_add.vhd -- Residual Addition + Post-LayerNorm for Transformer
-- Encoder Block
--
-- Computes: output = LayerNorm(input + residual)
--
-- This implements the Post-LN pattern: the residual (skip connection) is added
-- to the main-path input element-wise, and the sum is then normalized through
-- a LayerNorm component.
--
-- Generics:
--   DATA_WIDTH   : bit width of each data element (default 16)
--   VEC_SIZE     : model dimension / number of elements per vector (default 512)
--
-- Ports (accel library streaming conventions):
--   clk, rstn
--   i_data      : main-path input data
--   i_data_valid: data valid qualifier
--   i_data_last : end-of-packet marker
--   i_data_channel: channel identifier
--   i_residual  : residual / skip-connection input data
--   i_residual_valid
--   i_residual_last
--   i_residual_channel
--   o_data      : normalized output data
--   o_valid     : output valid qualifier
--   o_last      : end-of-packet marker
--   o_channel   : channel identifier
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.utilities.all;

entity residual_add is
  generic (
    DATA_WIDTH : positive := 16;
    VEC_SIZE   : positive := 512
  );
  port (
    -- Clock and reset
    clk   : in  std_logic;
    rstn : in  std_logic;

    -- Main-path input
    i_data         : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
    i_data_valid   : in  std_logic;
    i_data_last    : in  std_logic;
    i_data_channel : in  integer range 0 to max_size_x - 1;

    -- Residual (skip-connection) input
    i_residual         : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
    i_residual_valid   : in  std_logic;
    i_residual_last    : in  std_logic;
    i_residual_channel : in  integer range 0 to max_size_x - 1;

    -- Normalized output
    o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    o_valid   : out std_logic;
    o_last    : out std_logic;
    o_channel : out integer range 0 to max_size_x - 1;

    -- LayerNorm parameter loading (pass-through to layernorm)
    i_params_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
    i_params_valid : in  std_logic;
    i_params_addr  : in  std_logic_vector(9 downto 0);
    i_params_sel   : in  std_logic
  );
end entity residual_add;

architecture rtl of residual_add is

  ----------------------------------------------------------------------------
  -- LayerNorm component declaration
  ----------------------------------------------------------------------------
  component layernorm is
    generic (
      DATA_WIDTH : positive := 16;
      VEC_SIZE   : positive := 512
    );
    port (
      clk       : in  std_logic;
      rstn     : in  std_logic;
      i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
      i_valid   : in  std_logic;
      i_last    : in  std_logic;
      i_channel : in  integer range 0 to max_size_x - 1;
      o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
      o_valid   : out std_logic;
      o_last    : out std_logic;
      o_channel : out integer range 0 to max_size_x - 1
    );
  end component layernorm;

  ----------------------------------------------------------------------------
  -- FSM states
  ----------------------------------------------------------------------------
  type t_fsm_state is (IDLE, BUFFER_RESIDUAL, ADD_ELEMENTS, LAYERNORM, DONE);

  signal state      : t_fsm_state;
  signal next_state : t_fsm_state;

  ----------------------------------------------------------------------------
  -- Residual buffering / latching registers
  ----------------------------------------------------------------------------
  signal residual_buf_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal residual_buf_valid   : std_logic;
  signal residual_buf_last    : std_logic;
  signal residual_buf_channel : integer range 0 to max_size_x - 1;

  ----------------------------------------------------------------------------
  -- Main-path delayed registers (to align with buffered residual)
  ----------------------------------------------------------------------------
  signal main_data_d1    : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal main_valid_d1   : std_logic;
  signal main_last_d1    : std_logic;
  signal main_channel_d1 : integer range 0 to max_size_x - 1;

  ----------------------------------------------------------------------------
  -- Addition stage (combinational + registered output)
  ----------------------------------------------------------------------------
  signal sum_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal sum_valid   : std_logic;
  signal sum_last    : std_logic;
  signal sum_channel : integer range 0 to max_size_x - 1;

  ----------------------------------------------------------------------------
  -- LayerNorm interface signals
  ----------------------------------------------------------------------------
  signal ln_data_in    : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal ln_valid_in   : std_logic;
  signal ln_last_in    : std_logic;
  signal ln_channel_in : integer range 0 to max_size_x - 1;

  signal ln_data_out    : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal ln_valid_out   : std_logic;
  signal ln_last_out    : std_logic;
  signal ln_channel_out : integer range 0 to max_size_x - 1;

  ----------------------------------------------------------------------------
  -- Element counter for tracking vector positions
  ----------------------------------------------------------------------------
  signal elem_cnt : unsigned(integer(ceil(log2(real(VEC_SIZE)))) downto 0);

begin

  ----------------------------------------------------------------------------
  -- FSM: state register
  ----------------------------------------------------------------------------
  p_fsm_reg : process(clk, rstn) is
  begin
    if rstn = '0' then
      state <= IDLE;
    elsif rising_edge(clk) then
      state <= next_state;
    end if;
  end process p_fsm_reg;

  ----------------------------------------------------------------------------
  -- FSM: next-state logic
  ----------------------------------------------------------------------------
  p_fsm_next : process(state,
                       i_data_valid, i_data_last,
                       i_residual_valid, i_residual_last,
                       elem_cnt) is
  begin
    next_state <= state;

    case state is

      when IDLE =>
        -- Wait for both main-path and residual data to become valid
        if i_data_valid = '1' and i_residual_valid = '1' then
          next_state <= BUFFER_RESIDUAL;
        end if;

      when BUFFER_RESIDUAL =>
        -- One cycle to latch the residual; proceed to addition
        next_state <= ADD_ELEMENTS;

      when ADD_ELEMENTS =>
        -- Continue adding elements until end-of-packet on both paths
        if i_data_valid = '1' and i_data_last = '1' then
          next_state <= LAYERNORM;
        end if;

      when LAYERNORM =>
        -- Wait for LayerNorm to finish processing the entire vector
        if ln_valid_out = '1' and ln_last_out = '1' then
          next_state <= DONE;
        end if;

      when DONE =>
        -- Return to idle for the next vector
        next_state <= IDLE;

      when others =>
        next_state <= IDLE;

    end case;
  end process p_fsm_next;

  ----------------------------------------------------------------------------
  -- Residual latching (BUFFER_RESIDUAL state)
  -- Capture the residual input on entry to BUFFER_RESIDUAL so it aligns
  -- with the main-path data on the next cycle.
  ----------------------------------------------------------------------------
  p_residual_latch : process(clk) is
  begin
    if rising_edge(clk) then
      if state = IDLE and i_residual_valid = '1' then
        residual_buf_data    <= i_residual;
        residual_buf_valid   <= i_residual_valid;
        residual_buf_last    <= i_residual_last;
        residual_buf_channel <= i_residual_channel;
      elsif state = ADD_ELEMENTS and i_residual_valid = '1' then
        residual_buf_data    <= i_residual;
        residual_buf_valid   <= i_residual_valid;
        residual_buf_last    <= i_residual_last;
        residual_buf_channel <= i_residual_channel;
      else
        residual_buf_valid <= '0';
      end if;
    end if;
  end process p_residual_latch;

  ----------------------------------------------------------------------------
  -- Main-path delay registers (pipeline stage for alignment)
  ----------------------------------------------------------------------------
  p_main_delay : process(clk) is
  begin
    if rising_edge(clk) then
      if rstn = '0' then
        main_data_d1    <= (others => '0');
        main_valid_d1   <= '0';
        main_last_d1    <= '0';
        main_channel_d1 <= 0;
      else
        main_data_d1    <= i_data;
        main_valid_d1   <= i_data_valid;
        main_last_d1    <= i_data_last;
        main_channel_d1 <= i_data_channel;
      end if;
    end if;
  end process p_main_delay;

  ----------------------------------------------------------------------------
  -- Element-wise addition (combinational)
  ----------------------------------------------------------------------------
  sum_data <= std_logic_vector(
    signed(main_data_d1) + signed(residual_buf_data)
  );

  sum_valid   <= main_valid_d1 and residual_buf_valid;
  sum_last    <= main_last_d1;
  sum_channel <= main_channel_d1;

  ----------------------------------------------------------------------------
  -- Addition output register (pipeline stage before LayerNorm)
  ----------------------------------------------------------------------------
  p_sum_reg : process(clk) is
  begin
    if rising_edge(clk) then
      if rstn = '0' then
        ln_data_in    <= (others => '0');
        ln_valid_in   <= '0';
        ln_last_in    <= '0';
        ln_channel_in <= 0;
      else
        ln_data_in    <= sum_data;
        ln_valid_in   <= sum_valid;
        ln_last_in    <= sum_last;
        ln_channel_in <= sum_channel;
      end if;
    end if;
  end process p_sum_reg;

  ----------------------------------------------------------------------------
  -- Element counter (for debugging / monitoring vector position)
  ----------------------------------------------------------------------------
  p_elem_cnt : process(clk) is
  begin
    if rising_edge(clk) then
      if rstn = '0' then
        elem_cnt <= (others => '0');
      elsif state = IDLE then
        elem_cnt <= (others => '0');
      elsif state = ADD_ELEMENTS and sum_valid = '1' then
        if sum_last = '1' then
          elem_cnt <= (others => '0');
        else
          elem_cnt <= elem_cnt + 1;
        end if;
      end if;
    end if;
  end process p_elem_cnt;

  ----------------------------------------------------------------------------
  -- LayerNorm instance
  ----------------------------------------------------------------------------
  u_layernorm : layernorm
    generic map (
      DATA_WIDTH => DATA_WIDTH,
      VEC_SIZE   => VEC_SIZE
    )
    port map (
      clk       => clk,
      rstn     => rstn,
      i_data    => ln_data_in,
      i_valid   => ln_valid_in,
      i_last    => ln_last_in,
      i_channel => ln_channel_in,
      o_data    => ln_data_out,
      o_valid   => ln_valid_out,
      o_last    => ln_last_out,
      o_channel => ln_channel_out,
      i_params_data  => i_params_data,
      i_params_valid => i_params_valid,
      i_params_addr  => i_params_addr,
      i_params_sel   => i_params_sel
    );

  ----------------------------------------------------------------------------
  -- Output assignment (pass-through from LayerNorm)
  ----------------------------------------------------------------------------
  o_data    <= ln_data_out;
  o_valid   <= ln_valid_out;
  o_last    <= ln_last_out;
  o_channel <= ln_channel_out;

end architecture rtl;
