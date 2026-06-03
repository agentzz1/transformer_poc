# Build the minimal seg_test sanity design -> seg_test.bit
set SCRIPT_DIR [file dirname [info script]]
set POC_DIR    [file normalize "$SCRIPT_DIR/.."]
set WORK_DIR   [file join $::env(USERPROFILE) vivado_work]
set PART       "xc7a35tcpg236-1"

catch { close_project }
create_project seg_test "$WORK_DIR/seg_test_proj" -part $PART -force
set_property target_language VHDL [current_project]

add_files -norecurse "$POC_DIR/seg_test.vhd"
set_property file_type {VHDL 2008} [get_files "$POC_DIR/seg_test.vhd"]
add_files -fileset constrs_1 -norecurse "$POC_DIR/seg_test.xdc"

set_property top seg_test [current_fileset]
update_compile_order -fileset sources_1

synth_design -top seg_test -part $PART
opt_design
place_design
route_design
write_bitstream -force "$WORK_DIR/seg_test.bit"
puts "SEG_TEST_BITSTREAM_DONE: $WORK_DIR/seg_test.bit"
exit 0
