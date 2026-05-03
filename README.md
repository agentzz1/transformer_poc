# FPGA Transformer Encoder Block -- Proof-of-Concept (VHDL)

A synthesisable, streaming VHDL implementation of a single **Post-LayerNorm Transformer Encoder Block** targeting FPGA/ASIC inference.

| Parameter     | Default value |
|---------------|--------------|
| `DATA_WIDTH`  | 16-bit signed fixed-point |
| `MODEL_DIM`   | 512 |
| `NUM_HEADS`   | 8 |
| `HEAD_DIM`    | 64 |
| `HIDDEN_DIM`  | 2048 |
| `SEQ_LEN`     | 64 tokens |

## Architecture overview

The encoder block follows the classic Post-LN pattern:

```
Input -> [MHA + ResidualAdd + LayerNorm] -> [FFN + ResidualAdd + LayerNorm] -> Output
```

Data flows as a **streaming serial interface** (one element per clock cycle), compatible with the `accel` library conventions (`valid` / `last` / `channel`).

### File list

| File | Description |
|------|-------------|
| `clog2_pkg.vhd` | Utility package: `clog2(n)` ceiling-log2 function |
| `gemm_os.vhd` | Output-Stationary systolic-array GEMM engine (`pe_os` PE + top-level) |
| `scalar_ops.vhd` | GELU LUT + reciprocal-sqrt LUT (standalone scalar pipelines) |
| `softmax.vhd` | Numerically-stable streaming softmax (two-pass: scan-max -> exp -> norm) |
| `layernorm.vhd` | LayerNorm with gamma/beta parameter loading, 256-entry rsqrt LUT |
| `mha_controller.vhd` | Multi-Head Attention controller: Q/K/V projection, score computation, softmax, attention, output projection |
| `ffn.vhd` | Feed-Forward Network: FC1 (MODEL_DIM->HIDDEN_DIM), GELU, FC2 (HIDDEN_DIM->MODEL_DIM) |
| `residual_add.vhd` | Residual addition + LayerNorm wrapper (skip-connection buffer + add) |
| `control_unit.vhd` | Top-level FSM sequencing MHA -> Add+LN -> FFN -> Add+LN |
| `encoder_block.vhd` | Structural top-level: instantiates MHA, residual_add_1, FFN, residual_add_2, control_unit |
| `tb_encoder_block.vhd` | Self-checking testbench with LFSR stimulus, file I/O, assertions |

## How to compile and simulate

Requires [GHDL](https://github.com/ghdl/ghdl) (VHDL-2008 capable) and the `accel` support library.

```bash
cd transformer_poc

# Compile all sources
make compile

# Run testbench (generates mha_out.txt, ffn_out.txt, encoder_out.txt)
make run

# Full flow: compile + run
make all

# Clean build artefacts
make clean
```

Manual GHDL invocation (if `make` is unavailable):

```bash
mkdir -p work
ghdl -a --std=08 --workdir=work clog2_pkg.vhd
ghdl -a --std=08 --workdir=work gemm_os.vhd
ghdl -a --std=08 --workdir=work scalar_ops.vhd
ghdl -a --std=08 --workdir=work softmax.vhd
ghdl -a --std=08 --workdir=work layernorm.vhd
ghdl -a --std=08 --workdir=work mha_controller.vhd
ghdl -a --std=08 --workdir=work ffn.vhd
ghdl -a --std=08 --workdir=work residual_add.vhd
ghdl -a --std=08 --workdir=work control_unit.vhd
ghdl -a --std=08 --workdir=work encoder_block.vhd
ghdl -a --std=08 --workdir=work tb_encoder_block.vhd

ghdl -e --std=08 --workdir=work tb_encoder_block
ghdl -r --std=08 --workdir=work tb_encoder_block --wave=waveform.ghw
```

## Known limitations / TODOs

1. **Weight memories are external** -- `mha_controller` and `ffn` expose `w_*_addr / w_*_data / w_*_re` ports but no physical memory models are instantiated. A wrapper or testbench must provide BRAM/URAM blocks.
2. **Bias memories (`b1`, `b2`) in `ffn`** are likewise external.
3. **LayerNorm gamma/beta parameters** must be loaded via `i_params_*` before operation; there is no on-chip initialisation.
4. **Scalar quantisation** -- all arithmetic uses signed fixed-point (Q1.15 or similar). No explicit scaling/quantisation calibration is performed; overflow may occur for large activations.
5. **GELU in `ffn.vhd`** instantiates `psum_activation` from the `accel` library; a fallback LUT-based GELU inside `scalar_ops.vhd` exists but is not wired into `ffn`.
6. **Synthesis target** -- currently optimised for simulation correctness. Pipelining, BRAM inference, and timing closure have not been validated on hardware.
7. **No multi-encoder stacking** -- only a single encoder block is implemented. A deeper transformer would require a wrapper that chains multiple `encoder_block` instances with weight-banking.
8. **Done-signal generation** in `encoder_block.vhd` uses edge-detection on `o_last` streams; this may need refinement for robust one-cycle pulse timing under backpressure.
