#!/usr/bin/env bash
# Build + run end-to-end tb_basys3_top simulation with GHDL.
# Dumps pe/mha/y1/ffn/encoder _vhdl.txt for comparison against Python golden.
set -e
GHDL="$HOME/AppData/Local/Microsoft/WinGet/Packages/ghdl.ghdl.ucrt64.mcode_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/ghdl"
STD="--std=08"
W="--workdir=work_basys -P=work_basys"

rm -rf work_basys; mkdir -p work_basys
rm -f *_vhdl.txt

# unisim dummy must live in its own library named "unisim"
"$GHDL" -a $STD $W --work=unisim unisim_dummy.vhd

# design sources, dependency order
for f in clog2_pkg.vhd weights_pkg.vhd utilities.vhd \
         gemm_os.vhd gemm_os_adapter.vhd softmax.vhd scalar_ops.vhd \
         layernorm.vhd weight_mem.vhd gemm_mm.vhd psum_activation.vhd \
         mha_controller.vhd ffn.vhd residual_add.vhd control_unit.vhd \
         encoder_block.vhd patch_embed.vhd classifier.vhd basys3_top.vhd \
         tb_basys3_top.vhd ; do
  echo "[a] $f"
  "$GHDL" -a $STD $W $f
done

echo "[e] tb_basys3_top"
"$GHDL" -e $STD $W tb_basys3_top

echo "[r] running (this takes a few minutes due to UART timing)..."
"$GHDL" -r $STD $W tb_basys3_top --ieee-asserts=disable --stop-time=120ms > basys_sim.log 2>&1 || true
echo "[done] log -> basys_sim.log"
tail -5 basys_sim.log
wc -l *_vhdl.txt 2>/dev/null || true
