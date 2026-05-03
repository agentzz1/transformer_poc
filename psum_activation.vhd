--------------------------------------------------------------------------------
-- psum_activation.vhd -- Activation Wrapper for FFN Output Stream
--------------------------------------------------------------------------------
-- Simple registered pass-through wrapper that matches the accel library
-- psum_activation interface.  Used by ffn.vhd as the GELU stage.
--
-- For a PoC the activation is bypassed (direct pass-through).  A real
-- GELU can be dropped in later without changing the interface.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

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
        i_channel : in  integer range 0 to 511;

        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer range 0 to 511
    );
end entity psum_activation;

architecture rtl of psum_activation is

    signal running    : std_logic;
    signal elem_cnt   : integer range 0 to NUM_ELEMENTS - 1;
    signal data_reg   : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal valid_reg  : std_logic;
    signal last_reg   : std_logic;
    signal chan_reg   : integer range 0 to 511;
    signal done_reg   : std_logic;

begin

    ---------------------------------------------------------------------------
    -- Control FSM: start -> running -> done
    ---------------------------------------------------------------------------
    proc_ctrl : process(clk, rstn)
    begin
        if rstn = '0' then
            running  <= '0';
            elem_cnt <= 0;
            done_reg <= '0';
        elsif rising_edge(clk) then
            done_reg <= '0';

            if start = '1' then
                running  <= '1';
                elem_cnt <= 0;
            elsif running = '1' and i_valid = '1' then
                if i_last = '1' or elem_cnt = NUM_ELEMENTS - 1 then
                    running  <= '0';
                    done_reg <= '1';
                else
                    elem_cnt <= elem_cnt + 1;
                end if;
            end if;
        end if;
    end process proc_ctrl;

    done <= done_reg;

    ---------------------------------------------------------------------------
    -- Registered datapath (1-cycle latency)
    ---------------------------------------------------------------------------
    proc_pipe : process(clk, rstn)
    begin
        if rstn = '0' then
            data_reg  <= (others => '0');
            valid_reg <= '0';
            last_reg  <= '0';
            chan_reg  <= 0;
        elsif rising_edge(clk) then
            if running = '1' and i_valid = '1' then
                -- PoC: pass-through (bypass activation)
                -- For GELU, replace with signed GELU LUT here.
                data_reg  <= i_data;
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
