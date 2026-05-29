-- =============================================================================
-- tb_classifier.vhd  —  GHDL testbench for classifier component
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_classifier is
end entity tb_classifier;

architecture sim of tb_classifier is

    constant DATA_WIDTH : positive := 8;
    constant D_MODEL    : positive := 32;
    constant SEQ_LEN    : positive := 16;
    constant N_CLS      : positive := 10;
    
    signal clk   : std_logic := '0';
    signal rstn  : std_logic := '0';
    signal start : std_logic := '0';
    signal done  : std_logic;
    
    signal i_data    : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal i_valid   : std_logic := '0';
    signal i_last    : std_logic := '0';
    signal i_channel : integer := 0;
    
    signal o_class   : integer range 0 to N_CLS - 1;
    
    constant CLK_PERIOD : time := 10 ns;

begin

    -- Clock generator
    clk <= not clk after CLK_PERIOD / 2;

    -- UUT
    uut : entity work.classifier
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            D_MODEL    => D_MODEL,
            SEQ_LEN    => SEQ_LEN,
            N_CLS      => N_CLS
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            start     => start,
            done      => done,
            i_data    => i_data,
            i_valid   => i_valid,
            i_last    => i_last,
            i_channel => i_channel,
            o_class   => o_class
        );

    -- Stimulus process
    p_stim : process is
        file f_in      : text open read_mode is "encoder_out_mnist.txt";
        variable l     : line;
        variable val_i : integer;
        variable count : integer := 0;
    begin
        -- Reset
        rstn  <= '0';
        start <= '0';
        wait for 4 * CLK_PERIOD;
        rstn <= '1';
        wait for 2 * CLK_PERIOD;
        
        -- Start pulse (matching first pixel arrival in basys3_top)
        start <= '1';
        wait for CLK_PERIOD;
        start <= '0';
        
        -- Wait some cycles simulating frame transmission over UART
        wait for 100 * CLK_PERIOD;
        
        report "[TB] Starting to stream encoder outputs into classifier...";
        
        -- Read 512 elements from file and feed them
        while not endfile(f_in) loop
            readline(f_in, l);
            read(l, val_i);
            
            i_data    <= std_logic_vector(to_signed(val_i, DATA_WIDTH));
            i_valid   <= '1';
            i_channel <= count;
            
            -- Set i_last at the end of each token
            if (count + 1) mod D_MODEL = 0 then
                i_last <= '1';
            else
                i_last <= '0';
            end if;
            
            wait for CLK_PERIOD;
            count := count + 1;
        end loop;
        
        -- Clear inputs
        i_valid   <= '0';
        i_last    <= '0';
        i_channel <= 0;
        
        report "[TB] Stream complete. Total elements sent: " & integer'image(count);
        
        -- Wait for done
        wait until done = '1';
        wait for CLK_PERIOD;
        
        report "[SUCCESS] Classifier finished! Predicted class: " & integer'image(o_class);
        
        wait for 10 * CLK_PERIOD;
        assert false report "Simulation finished successfully" severity failure;
        wait;
    end process p_stim;

end architecture sim;
