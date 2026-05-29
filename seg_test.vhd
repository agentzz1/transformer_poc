-- =============================================================================
-- seg_test.vhd  --  Minimal Basys 3 sanity design: show "run" on 7-segment
-- =============================================================================
-- No MMCM, no UART, no logic -- just a refresh counter that multiplexes the
-- four 7-segment digits to display  r u n  (+ a blinking LED bar).
-- Purpose: prove the board powers up (DONE = HIGH) and the bitstream loads.
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity seg_test is
    port (
        clk : in  std_logic;                       -- 100 MHz, pin W5
        seg : out std_logic_vector(6 downto 0);    -- {g,f,e,d,c,b,a}, active-low
        dp  : out std_logic;                       -- decimal point, active-low
        an  : out std_logic_vector(3 downto 0);    -- digit anodes, active-low
        led : out std_logic_vector(3 downto 0)     -- status LEDs
    );
end entity seg_test;

architecture rtl of seg_test is
    signal cnt       : unsigned(26 downto 0) := (others => '0');
    signal digit_sel : unsigned(1 downto 0);

    -- 7-seg patterns, active-low (0 = segment ON), bit order seg(6..0)=(g,f,e,d,c,b,a)
    constant CH_R     : std_logic_vector(6 downto 0) := "0101111"; -- r  (e,g)
    constant CH_U     : std_logic_vector(6 downto 0) := "1100011"; -- u  (c,d,e)
    constant CH_N     : std_logic_vector(6 downto 0) := "0101011"; -- n  (c,e,g)
    constant CH_BLANK : std_logic_vector(6 downto 0) := "1111111"; -- blank
begin

    -- Free-running counter off the 100 MHz clock
    process (clk) is
    begin
        if rising_edge(clk) then
            cnt <= cnt + 1;
        end if;
    end process;

    -- ~760 Hz per-digit refresh (100 MHz / 2^17)
    digit_sel <= cnt(17 downto 16);

    -- Digit multiplexer: display  r u n _   (leftmost = an[3])
    process (digit_sel) is
    begin
        case digit_sel is
            when "11"   => an <= "0111"; seg <= CH_R;      -- leftmost = r
            when "10"   => an <= "1011"; seg <= CH_U;      --           u
            when "01"   => an <= "1101"; seg <= CH_N;      --           n
            when others => an <= "1110"; seg <= CH_BLANK;  -- rightmost = blank
        end case;
    end process;

    dp  <= '1';                                            -- decimal point off
    led <= std_logic_vector(cnt(26 downto 23));            -- visible blink (~1.5 Hz)

end architecture rtl;
