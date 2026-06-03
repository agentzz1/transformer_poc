# =============================================================================
# flash_basys3.tcl  --  Program basys3_top.bit onto the Basys 3 over JTAG
# =============================================================================
# Mirrors the known-good flash.tcl that reached "startup status: HIGH" on this
# exact board, plus a short poll so a freshly re-plugged USB target is found.
# =============================================================================

set BIT [file join $::env(USERPROFILE) vivado_work basys3_top.bit]
if {![file exists $BIT]} {
    puts "ERROR: bitstream not found: $BIT"
    exit 1
}

open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag

# Freshly re-plugged USB devices take a few seconds to enumerate.
set targets {}
for {set i 0} {$i < 20} {incr i} {
    catch { refresh_hw_server }
    set targets [get_hw_targets -quiet]
    if {[llength $targets] > 0} { break }
    puts "Waiting for JTAG target... ($i)"
    after 1000
}
if {[llength $targets] == 0} {
    puts "ERROR: no JTAG target found -- is the Basys 3 plugged in and powered on?"
    exit 1
}
puts "Found target(s): $targets"

open_hw_target
current_hw_device [get_hw_devices xc7a35t_0]
set_property PROGRAM.FILE $BIT [get_hw_devices xc7a35t_0]
program_hw_devices [get_hw_devices xc7a35t_0]

puts "=========================================="
puts " Basys 3 programmed with $BIT"
puts "=========================================="
close_hw_target
disconnect_hw_server
close_hw_manager
exit 0
