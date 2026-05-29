-- =============================================================================
-- tb_basys3_top.vhd  —  End-to-End Testbench for basys3_top (capturing encoder output)
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_basys3_top is
end entity tb_basys3_top;

architecture sim of tb_basys3_top is

    signal clk      : std_logic := '0';
    signal btnc     : std_logic := '0';
    signal uart_rxd : std_logic := '1';
    signal uart_txd : std_logic;
    signal seg      : std_logic_vector(6 downto 0);
    signal dp       : std_logic;
    signal an       : std_logic_vector(3 downto 0);
    signal led      : std_logic_vector(3 downto 0);

    constant CLK_PERIOD : time := 10 ns; -- 100 MHz input clock
    constant BIT_PERIOD : time := 8.68 us; -- 115200 Baud (approx 8.6805 us)

    -- Capture UART-TX from FPGA
    signal tx_byte : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_done : std_logic := '0';

begin

    -- Clock generator (100 MHz)
    clk <= not clk after CLK_PERIOD / 2;

    -- UUT
    uut : entity work.basys3_top
        port map (
            clk      => clk,
            btnc     => btnc,
            uart_rxd => uart_rxd,
            uart_txd => uart_txd,
            seg      => seg,
            dp       => dp,
            an       => an,
            led      => led
        );

    -- LED Monitor
    p_led_monitor : process(clk) is
        variable old_led : std_logic_vector(3 downto 0) := "0000";
    begin
        if rising_edge(clk) then
            if led /= old_led then
                report "[LED MONITOR] LED changed: " & 
                       std_logic'image(led(3)) & std_logic'image(led(2)) &
                       std_logic'image(led(1)) & std_logic'image(led(0)) &
                       " at " & time'image(now) severity note;
                old_led := led;
            end if;
        end if;
    end process p_led_monitor;

    -- Capture Patch Embed Output to a file
    p_pe_capture : process is
        file out_file      : text open write_mode is "pe_out_vhdl.txt";
        variable l         : line;
        variable count     : integer := 0;
        alias pe_o_valid is << signal .tb_basys3_top.uut.pe_o_valid : std_logic >>;
        alias pe_o_data  is << signal .tb_basys3_top.uut.pe_o_data : std_logic_vector(7 downto 0) >>;
        alias clk20 is << signal .tb_basys3_top.uut.clk20 : std_logic >>;
    begin
        loop
            wait until rising_edge(clk20);
            if pe_o_valid = '1' then
                write(l, to_integer(signed(pe_o_data)));
                writeline(out_file, l);
                count := count + 1;
                if count = 512 then
                    report "[CAPTURE] Captured all 512 Patch Embed output elements to pe_out_vhdl.txt" severity note;
                    exit;
                end if;
            end if;
        end loop;
        wait;
    end process p_pe_capture;

    -- Capture MHA Output to a file
    p_mha_capture : process is
        file out_file      : text open write_mode is "mha_out_vhdl.txt";
        variable l         : line;
        variable count     : integer := 0;
        alias enc_mha_valid is << signal .tb_basys3_top.uut.enc_mha_valid : std_logic >>;
        alias enc_mha_data  is << signal .tb_basys3_top.uut.enc_mha_data : std_logic_vector(7 downto 0) >>;
        alias clk20 is << signal .tb_basys3_top.uut.clk20 : std_logic >>;
    begin
        loop
            wait until rising_edge(clk20);
            if enc_mha_valid = '1' then
                write(l, to_integer(signed(enc_mha_data)));
                writeline(out_file, l);
                count := count + 1;
                if count = 512 then
                    report "[CAPTURE] Captured all 512 MHA output elements to mha_out_vhdl.txt" severity note;
                    exit;
                end if;
            end if;
        end loop;
        wait;
    end process p_mha_capture;

    -- Capture LayerNorm1 Output to a file
    p_y1_capture : process is
        file out_file      : text open write_mode is "y1_out_vhdl.txt";
        variable l         : line;
        variable count     : integer := 0;
        alias res1_out_valid is << signal .tb_basys3_top.uut.u_encoder.res1_out_valid : std_logic >>;
        alias res1_out_data  is << signal .tb_basys3_top.uut.u_encoder.res1_out_data : std_logic_vector(7 downto 0) >>;
        alias clk20 is << signal .tb_basys3_top.uut.clk20 : std_logic >>;
    begin
        loop
            wait until rising_edge(clk20);
            if res1_out_valid = '1' then
                write(l, to_integer(signed(res1_out_data)));
                writeline(out_file, l);
                count := count + 1;
                if count = 512 then
                    report "[CAPTURE] Captured all 512 LayerNorm1 output elements to y1_out_vhdl.txt" severity note;
                    exit;
                end if;
            end if;
        end loop;
        wait;
    end process p_y1_capture;

    -- Capture FFN Output to a file
    p_ffn_capture : process is
        file out_file      : text open write_mode is "ffn_out_vhdl.txt";
        variable l         : line;
        variable count     : integer := 0;
        alias enc_ffn_valid is << signal .tb_basys3_top.uut.enc_ffn_valid : std_logic >>;
        alias enc_ffn_data  is << signal .tb_basys3_top.uut.enc_ffn_data : std_logic_vector(7 downto 0) >>;
        alias clk20 is << signal .tb_basys3_top.uut.clk20 : std_logic >>;
    begin
        loop
            wait until rising_edge(clk20);
            if enc_ffn_valid = '1' then
                write(l, to_integer(signed(enc_ffn_data)));
                writeline(out_file, l);
                count := count + 1;
                if count = 512 then
                    report "[CAPTURE] Captured all 512 FFN output elements to ffn_out_vhdl.txt" severity note;
                    exit;
                end if;
            end if;
        end loop;
        wait;
    end process p_ffn_capture;

    -- Capture Encoder Output to a file
    p_enc_capture : process is
        file out_file      : text open write_mode is "encoder_out_vhdl.txt";
        variable l         : line;
        variable count     : integer := 0;
        alias enc_o_valid is << signal .tb_basys3_top.uut.enc_o_valid : std_logic >>;
        alias enc_o_data  is << signal .tb_basys3_top.uut.enc_o_data : std_logic_vector(7 downto 0) >>;
        alias clk20 is << signal .tb_basys3_top.uut.clk20 : std_logic >>;
    begin
        loop
            wait until rising_edge(clk20);
            if enc_o_valid = '1' then
                write(l, to_integer(signed(enc_o_data)));
                writeline(out_file, l);
                count := count + 1;
                if count = 512 then
                    report "[CAPTURE] Captured all 512 encoder output elements to encoder_out_vhdl.txt" severity note;
                    exit;
                end if;
            end if;
        end loop;
        wait;
    end process p_enc_capture;

    -- Monitor JTAG/UART-TX output from FPGA
    p_uart_tx_rx : process is
        variable bit_cnt : integer := 0;
        variable shift   : std_logic_vector(7 downto 0) := (others => '0');
    begin
        tx_done <= '0';
        -- Wait for start bit
        wait until uart_txd = '0';
        -- Wait to center of start bit
        wait for BIT_PERIOD / 2;
        if uart_txd = '0' then
            -- Sample 8 data bits
            for i in 0 to 7 loop
                wait for BIT_PERIOD;
                shift(i) := uart_txd;
            end loop;
            -- Wait for stop bit
            wait for BIT_PERIOD;
            tx_byte <= shift;
            tx_done <= '1';
            report "[MONITOR] Received byte from FPGA: " & integer'image(to_integer(unsigned(shift))) severity note;
            wait for CLK_PERIOD;
        end if;
    end process p_uart_tx_rx;

    -- Stimulus process
    p_stim : process is
        file f_in      : text open read_mode is "pixels_raw.txt";
        variable l     : line;
        variable val_i : integer;
        variable count : integer := 0;
        variable char_val : std_logic_vector(7 downto 0);
    begin
        -- Assert Reset
        btnc <= '1';
        wait for 200 ns;
        btnc <= '0';
        
        -- Wait for MMCM lock and reset sync (typically within a few microseconds)
        wait for 10 us;
        
        report "[TB] Starting end-to-end UART transmission of 784 pixels...";
        
        -- Read 784 pixels from pixels_raw.txt and send them over UART
        while not endfile(f_in) loop
            readline(f_in, l);
            read(l, val_i);
            
            char_val := std_logic_vector(to_unsigned(val_i, 8));
            
            -- UART Start bit
            uart_rxd <= '0';
            wait for BIT_PERIOD;
            
            -- 8 data bits (LSB-first)
            for i in 0 to 7 loop
                uart_rxd <= char_val(i);
                wait for BIT_PERIOD;
            end loop;
            
            -- Stop bit
            uart_rxd <= '1';
            wait for BIT_PERIOD;
            
            count := count + 1;
            
            -- Small inter-byte delay
            wait for 1 us;
        end loop;
        
        report "[TB] Sent 784 bytes. Total pixels sent: " & integer'image(count);
        
        -- Wait for FPGA responses
        -- First response should be 0xA5 (165) which is the frame-received ACK
        wait until tx_done = '1';
        report "[TB] Received first response byte: " & integer'image(to_integer(unsigned(tx_byte)));
        assert to_integer(unsigned(tx_byte)) = 165
            report "[FAIL] First byte is not 0xA5 ACK! Got: " & integer'image(to_integer(unsigned(tx_byte)))
            severity error;
            
        -- Second response should be the predicted class (0-9)
        wait until tx_done = '1';
        report "[SUCCESS] Predicted class byte received: " & integer'image(to_integer(unsigned(tx_byte)));
        
        wait for 100 us;
        assert false report "End-to-End Simulation Finished Successfully" severity failure;
        wait;
    end process p_stim;

end architecture sim;
