# =============================================================================
# basys3_impl.tcl  --  Full synthesis + implementation + bitstream for Basys 3
# =============================================================================
# Usage (from the vivado_synth_test directory):
#   vivado -mode batch -source basys3_impl.tcl
#
# Outputs (inside WORK_DIR):
#   util_basys3.rpt      - resource utilisation after synthesis
#   timing_basys3.rpt    - timing summary after implementation
#   basys3_top.bit       - bitstream ready to program to Basys 3
# =============================================================================

set SCRIPT_DIR [file dirname [info script]]
set POC_DIR    [file normalize "$SCRIPT_DIR/.."]
# Place the Vivado project on a local (non-OneDrive) path to avoid file-lock
# issues from OneDrive sync.  Change WORK_DIR if needed.
set WORK_DIR   "C:/Users/maogo/vivado_work"
set PROJ_NAME  "basys3_impl"
set PROJ_DIR   $WORK_DIR
set PART       "xc7a35tcpg236-1"

# --------------------------------------------------------------------------
# Create / overwrite project
# --------------------------------------------------------------------------
catch { close_project }
create_project $PROJ_NAME "$PROJ_DIR/$PROJ_NAME" -part $PART -force
set_property target_language VHDL [current_project]

# --------------------------------------------------------------------------
# Source files (compile order)
# --------------------------------------------------------------------------
set src_files [list \
    "$POC_DIR/clog2_pkg.vhd"          \
    "$POC_DIR/utilities.vhd"           \
    "$POC_DIR/scalar_ops.vhd"          \
    "$POC_DIR/gemm_os.vhd"             \
    "$POC_DIR/gemm_os_adapter.vhd"     \
    "$POC_DIR/gemm_mm.vhd"             \
    "$POC_DIR/softmax.vhd"             \
    "$POC_DIR/layernorm.vhd"           \
    "$POC_DIR/psum_activation.vhd"     \
    "$POC_DIR/weights_pkg.vhd"         \
    "$POC_DIR/weight_mem.vhd"          \
    "$POC_DIR/residual_add.vhd"        \
    "$POC_DIR/mha_controller.vhd"      \
    "$POC_DIR/ffn.vhd"                 \
    "$POC_DIR/control_unit.vhd"        \
    "$POC_DIR/encoder_block.vhd"       \
    "$POC_DIR/patch_embed.vhd"         \
    "$POC_DIR/classifier.vhd"          \
    "$POC_DIR/basys3_top.vhd"          \
]

add_files -norecurse $src_files
foreach f $src_files {
    set_property file_type {VHDL 2008} [get_files $f]
}

# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------
add_files -fileset constrs_1 -norecurse "$POC_DIR/basys3_constraints.xdc"

# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
set_property top basys3_top [current_fileset]
update_compile_order -fileset sources_1

# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------
puts "=========================================="
puts " Step 1: Synthesis"
puts "=========================================="
set t0 [clock seconds]
if {[catch {
    synth_design -top basys3_top -part $PART
} err]} {
    puts "ERROR in synthesis: $err"
    exit 1
}
set dt [expr {[clock seconds] - $t0}]
puts "Synthesis done in $dt s"
report_utilization -file "$PROJ_DIR/util_basys3.rpt"
report_timing_summary -file "$PROJ_DIR/timing_synth_basys3.rpt"
puts "Utilisation report: $PROJ_DIR/util_basys3.rpt"

# --------------------------------------------------------------------------
# Implementation (place & route)
# --------------------------------------------------------------------------
puts "=========================================="
puts " Step 2: Implementation (P&R)"
puts "=========================================="
set t0 [clock seconds]
if {[catch {
    opt_design
    place_design
    route_design
} err]} {
    puts "ERROR in implementation: $err"
    exit 1
}
set dt [expr {[clock seconds] - $t0}]
puts "Implementation done in $dt s"
report_timing_summary -file "$PROJ_DIR/timing_impl_basys3.rpt"
report_utilization    -file "$PROJ_DIR/util_impl_basys3.rpt"

# --------------------------------------------------------------------------
# Bitstream generation
# --------------------------------------------------------------------------
puts "=========================================="
puts " Step 3: Bitstream"
puts "=========================================="
set t0 [clock seconds]
if {[catch {
    write_bitstream -force "$PROJ_DIR/basys3_top.bit"
} err]} {
    puts "ERROR in bitstream: $err"
    exit 1
}
set dt [expr {[clock seconds] - $t0}]
puts "Bitstream written in $dt s"
puts "Output: $PROJ_DIR/basys3_top.bit"

puts ""
puts "=========================================="
puts " All steps complete."
puts "=========================================="
exit 0
