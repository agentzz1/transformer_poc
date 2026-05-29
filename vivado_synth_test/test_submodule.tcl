# =============================================================================
# test_submodule.tcl -- Synthesize a single POC submodule (out-of-context)
# =============================================================================
# Usage:
#   vivado -mode batch -source test_submodule.tcl -tclargs <module_name>
#
# module_name = layernorm | softmax | mha_controller | ffn | encoder_block
#             | patch_embed | classifier | basys3_top
# =============================================================================

if {[llength $argv] < 1} {
    puts "ERROR: pass module name as -tclargs <module>"
    exit 1
}
set MOD [lindex $argv 0]

set PROJ_DIR [file dirname [info script]]
set POC_DIR  [file normalize "$PROJ_DIR/.."]
set PROJ_NAME "synth_${MOD}"

catch { close_project }
create_project $PROJ_NAME "$PROJ_DIR/$PROJ_NAME" -part xc7a35tcpg236-1 -force
set_property target_language VHDL [current_project]

# All POC sources (compile order)
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

# For encoder_block, use synth_wrapper as top to force structural architecture
# (encoder_block.vhd also contains sim_reference which uses ieee.math_real)
set TOP_MOD $MOD
if {$MOD eq "encoder_block"} {
    lappend src_files "$PROJ_DIR/synth_wrapper.vhd"
    set TOP_MOD "synth_wrapper"
}

add_files -norecurse $src_files
foreach f $src_files { set_property file_type {VHDL 2008} [get_files $f] }

set_property top $TOP_MOD [current_fileset]
update_compile_order -fileset sources_1

# Generics for MNIST int8 (override defaults)
set gens "DATA_WIDTH=8"
switch $MOD {
    "layernorm"  { append gens " VEC_SIZE=32" }
    "softmax"    { append gens " SEQ_LEN=16" }
    "mha_controller" { append gens " MODEL_DIM=32 NUM_HEADS=1 HEAD_DIM=32 SEQ_LEN=16" }
    "ffn"          { append gens " MODEL_DIM=32 HIDDEN_DIM=64 SEQ_LEN=16" }
    "patch_embed"  { append gens " PATCH_SIZE=7 IMG_SIZE=28 D_MODEL=32 SEQ_LEN=16" }
    "classifier"   { append gens " D_MODEL=32 SEQ_LEN=16 N_CLS=10" }
    "basys3_top"   {
        # No generics -- all dimensions are fixed in the top-level entity
        set gens ""
    }
    "encoder_block" {
        # synth_wrapper has no generics — dimensions are hardcoded in the wrapper
        set gens ""
    }
}
if {$gens ne ""} {
    set_property generic $gens [current_fileset]
}

# OOC (out-of-context) synthesis -- no IOBs, just core logic
puts "=========================================="
puts " OOC synthesis of $MOD with generics: $gens"
puts "=========================================="

set t0 [clock seconds]
if {[catch {
    synth_design -top $TOP_MOD -part xc7a35tcpg236-1 -mode out_of_context
} err]} {
    puts "ERROR during synth_design: $err"
    exit 1
}
set dt [expr {[clock seconds] - $t0}]
puts "=========================================="
puts " $MOD synth OK in $dt seconds"
puts "=========================================="
report_utilization -file "$PROJ_DIR/util_${MOD}.rpt"
puts "Report: $PROJ_DIR/util_${MOD}.rpt"
exit 0
