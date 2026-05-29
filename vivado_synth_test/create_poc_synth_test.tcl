# =============================================================================
# create_poc_synth_test.tcl  --  Vivado Synthesis Test for POC Encoder Block
# =============================================================================
# Target:  Basys 3 / Artix-7 35T  (xc7a35tcpg236-1)
# Purpose: Check that the POC encoder_block synthesizes cleanly with
#          DATA_WIDTH=8 and MNIST dimensions.
#
# Usage (Vivado Tcl Console or batch):
#   source C:/Users/maogo/OneDrive/transformer/transformer_poc/vivado_synth_test/create_poc_synth_test.tcl
# =============================================================================

set PROJ_DIR  [file dirname [info script]]
set POC_DIR   [file normalize "$PROJ_DIR/.."]
set PROJ_NAME "poc_synth_test"

# Close any open project first
catch { close_project }

# ---------------------------------------------------------------------------
# Create project (synthesis only, no board part needed)
# ---------------------------------------------------------------------------
create_project $PROJ_NAME "$PROJ_DIR/$PROJ_NAME" -part xc7a35tcpg236-1 -force

set_property target_language VHDL [current_project]

# ---------------------------------------------------------------------------
# Add source files  (compile order matters: packages first)
# ---------------------------------------------------------------------------
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
    "$POC_DIR/weight_mem.vhd"          \
    "$POC_DIR/residual_add.vhd"        \
    "$POC_DIR/mha_controller.vhd"      \
    "$POC_DIR/ffn.vhd"                 \
    "$POC_DIR/control_unit.vhd"        \
    "$POC_DIR/encoder_block.vhd"       \
    "$PROJ_DIR/synth_wrapper.vhd"      \
]

add_files -norecurse $src_files

# ---------------------------------------------------------------------------
# Set VHDL-2008 for all source files
# (POC uses process(all), integer types in ports, etc.)
# ---------------------------------------------------------------------------
foreach f $src_files {
    set_property file_type {VHDL 2008} [get_files $f]
}

# ---------------------------------------------------------------------------
# Set top-level entity
# ---------------------------------------------------------------------------
set_property top synth_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---------------------------------------------------------------------------
# Synthesis settings
# ---------------------------------------------------------------------------
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt   [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.KEEP_EQUIVALENT_REGISTERS 1 [get_runs synth_1]

# ---------------------------------------------------------------------------
# Run synthesis
# ---------------------------------------------------------------------------
puts "========================================================"
puts " Starting synthesis for POC encoder_block (DATA_WIDTH=8)"
puts " Target: xc7a35tcpg236-1 (Basys 3 / Artix-7 35T)"
puts "========================================================"

launch_runs synth_1 -jobs 4
wait_on_run synth_1

set synth_status [get_property STATUS [get_runs synth_1]]
puts ""
puts "========================================================"
puts " Synthesis status: $synth_status"
puts "========================================================"

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
open_run synth_1 -name synth_1

report_utilization -file "$PROJ_DIR/utilization_synth.rpt"
report_timing_summary -file "$PROJ_DIR/timing_synth.rpt" -max_paths 10
report_methodology -file "$PROJ_DIR/methodology_synth.rpt"

puts ""
puts "Reports written to: $PROJ_DIR"
puts "  utilization_synth.rpt"
puts "  timing_synth.rpt"
puts "  methodology_synth.rpt"
puts ""
puts "Check the Tcl console and Messages tab for any CRITICAL WARNING or ERROR."

# =============================================================================
# End of file create_poc_synth_test.tcl
# =============================================================================
