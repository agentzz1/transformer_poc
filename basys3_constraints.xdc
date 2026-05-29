## =============================================================================
## basys3_constraints.xdc  --  Pin/clock constraints for MNIST ViT on Basys 3
## Digilent Basys 3  (Artix-7 XC7A35T-1CPG236C)
## =============================================================================

## 100 MHz crystal oscillator (W5) -- input to MMCM only
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk -period 10.000 -waveform {0 5} [get_ports clk]

## MMCM-generated 20 MHz clock (50 ns period).
## Vivado auto-derives this constraint from the MMCME2_BASE generic map, but
## we add it explicitly so timing reports clearly show the 20 MHz domain.
## The MMCM output pin path: u_mmcm/CLKOUT0 -> u_clk20_buf/O (BUFG)
create_generated_clock -add -name clk20 \
    -master_clock [get_clocks sys_clk] \
    -source [get_ports clk] \
    -multiply_by 1 -divide_by 5 \
    [get_pins u_clk20_buf/O]

## BTNC centre push-button -> active-high reset  (U18)
set_property PACKAGE_PIN U18 [get_ports btnc]
set_property IOSTANDARD LVCMOS33 [get_ports btnc]

## USB-UART bridge RX (data from PC to FPGA, B18)
set_property PACKAGE_PIN B18 [get_ports uart_rxd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rxd]

## USB-UART bridge TX (data from FPGA to PC, A18)
set_property PACKAGE_PIN A18 [get_ports uart_txd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_txd]

## USB-UART bridge TX (data from FPGA to PC, A18)
set_property PACKAGE_PIN A18 [get_ports uart_txd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_txd]

## 7-Segment Display -- cathodes, active-low
##   seg[0]=CA, seg[1]=CB, seg[2]=CC, seg[3]=CD
##   seg[4]=CE, seg[5]=CF, seg[6]=CG
set_property PACKAGE_PIN W7  [get_ports {seg[0]}] ;# CA (segment a)
set_property IOSTANDARD LVCMOS33 [get_ports {seg[0]}]

set_property PACKAGE_PIN W6  [get_ports {seg[1]}] ;# CB (segment b)
set_property IOSTANDARD LVCMOS33 [get_ports {seg[1]}]

set_property PACKAGE_PIN U8  [get_ports {seg[2]}] ;# CC (segment c)
set_property IOSTANDARD LVCMOS33 [get_ports {seg[2]}]

set_property PACKAGE_PIN V8  [get_ports {seg[3]}] ;# CD (segment d)
set_property IOSTANDARD LVCMOS33 [get_ports {seg[3]}]

set_property PACKAGE_PIN U5  [get_ports {seg[4]}] ;# CE (segment e)
set_property IOSTANDARD LVCMOS33 [get_ports {seg[4]}]

set_property PACKAGE_PIN V5  [get_ports {seg[5]}] ;# CF (segment f)
set_property IOSTANDARD LVCMOS33 [get_ports {seg[5]}]

set_property PACKAGE_PIN U7  [get_ports {seg[6]}] ;# CG (segment g)
set_property IOSTANDARD LVCMOS33 [get_ports {seg[6]}]

## Decimal point
set_property PACKAGE_PIN V7  [get_ports dp]
set_property IOSTANDARD LVCMOS33 [get_ports dp]

## 7-Segment digit anodes, active-low  (an[0] = rightmost digit)
set_property PACKAGE_PIN U2  [get_ports {an[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an[0]}]

set_property PACKAGE_PIN U4  [get_ports {an[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an[1]}]

set_property PACKAGE_PIN V4  [get_ports {an[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an[2]}]

set_property PACKAGE_PIN W4  [get_ports {an[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an[3]}]

## Status LEDs  (active-high)
set_property PACKAGE_PIN U16 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]

set_property PACKAGE_PIN E19 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]

set_property PACKAGE_PIN U19 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]

set_property PACKAGE_PIN V19 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]

## =============================================================================
## Timing constraints
## =============================================================================
## Treat UART RX/TX as false path for static timing (much slower than clock)
set_false_path -from [get_ports uart_rxd]
set_false_path -to   [get_ports uart_txd]

## Treat button inputs as false path (async, debounced in logic)
set_false_path -from [get_ports btnc]

## =============================================================================
## Configuration
## =============================================================================
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
## NOTE: CONFIG_MODE SPIx4 / SPI_BUSWIDTH are only needed for booting the
## design from the on-board SPI flash.  They reconfigure the startup sequence
## for SPI-master boot, which makes a *volatile JTAG* download fail with
## "End of startup status: LOW".  The known-good board_top flash had these
## OFF.  Re-enable only if you later program the SPI flash for standalone boot.
# set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
# set_property CONFIG_MODE SPIx4 [current_design]

## Bank configuration voltage -- REQUIRED for the FPGA to reach DONE
## (startup status HIGH) when programmed over JTAG.  Without these, JTAG
## programming aborts with "End of startup status: LOW".  Earlier successful
## flashes in this project had these set; the Basys 3 config bank is 3.3 V.
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
