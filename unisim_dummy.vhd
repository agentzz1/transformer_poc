-- =============================================================================
-- unisim_dummy.vhd  —  Dummy Xilinx Unisim library components for GHDL simulation
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;

package vcomponents is
    component MMCME2_BASE is
        generic (
            BANDWIDTH        : string := "OPTIMIZED";
            CLKFBOUT_MULT_F  : real := 5.0;
            CLKIN1_PERIOD    : real := 10.0;
            CLKOUT0_DIVIDE_F : real := 5.0;
            DIVCLK_DIVIDE    : integer := 1;
            CLKFBOUT_PHASE   : real := 0.0;
            CLKOUT0_DUTY_CYCLE : real := 0.5;
            CLKOUT0_PHASE    : real := 0.0;
            REF_JITTER1      : real := 0.01;
            STARTUP_WAIT     : boolean := false
        );
        port (
            CLKIN1   : in  std_logic;
            CLKFBIN  : in  std_logic;
            CLKFBOUT : out std_logic;
            CLKOUT0  : out std_logic;
            LOCKED   : out std_logic;
            RST      : in  std_logic;
            PWRDWN   : in  std_logic
        );
    end component;

    component BUFG is
        port (
            I : in  std_logic;
            O : out std_logic
        );
    end component;
end package vcomponents;

library ieee;
use ieee.std_logic_1164.all;

entity MMCME2_BASE is
    generic (
        BANDWIDTH        : string := "OPTIMIZED";
        CLKFBOUT_MULT_F  : real := 5.0;
        CLKIN1_PERIOD    : real := 10.0;
        CLKOUT0_DIVIDE_F : real := 5.0;
        DIVCLK_DIVIDE    : integer := 1;
        CLKFBOUT_PHASE   : real := 0.0;
        CLKOUT0_DUTY_CYCLE : real := 0.5;
        CLKOUT0_PHASE    : real := 0.0;
        REF_JITTER1      : real := 0.01;
        STARTUP_WAIT     : boolean := false
    );
    port (
        CLKIN1   : in  std_logic;
        CLKFBIN  : in  std_logic;
        CLKFBOUT : out std_logic;
        CLKOUT0  : out std_logic;
        LOCKED   : out std_logic;
        RST      : in  std_logic;
        PWRDWN   : in  std_logic
    );
end entity MMCME2_BASE;

architecture dummy of MMCME2_BASE is
    signal clk20_int : std_logic := '0';
begin
    -- Simple, robust, non-blocking 20 MHz clock generation (50 ns period)
    clk20_int <= not clk20_int after 25 ns;
    
    -- Locked signal goes high after a brief delay
    LOCKED   <= '0', '1' after 100 ns;
    
    CLKFBOUT <= CLKIN1;
    CLKOUT0  <= clk20_int;
end architecture dummy;

library ieee;
use ieee.std_logic_1164.all;

entity BUFG is
    port (
        I : in  std_logic;
        O : out std_logic
    );
end entity BUFG;

architecture dummy of BUFG is
begin
    O <= I;
end architecture dummy;
