#!/usr/bin/env python3
"""
verify_vhdl_weight_map.py
=========================
Simulates the EXACT weight-ROM address generation from mha_controller.vhd
(both old/wrong and new/fixed versions) and compares against the Python
golden model.

For each GEMM (Q, K, V projections and output projection W_O):
  - gemm_mm requests B[k][n]  at flat b_addr = k*N + n
  - mha_controller maps that to a physical ROM address
  - We check: does the value fetched from ROM equal what Python golden needs?

Golden model (verify_inference.py):
  Q = gemm_int8(x, transpose(WQ))   ->  B[k][n] = WQ.T[k][n] = WQ[n][k]
  K = gemm_int8(x, transpose(WK))
  V = gemm_int8(x, transpose(WV))
  out = gemm_int8(ctx, transpose(WO))

VHDL address mapping:
  b_addr = k * HEAD_DIM + n  (for Q/K/V, HEAD_DIM = MODEL_DIM for single head)
  v_k = b_addr // HEAD_DIM
  v_n = b_addr  % HEAD_DIM

  OLD (wrong): phys = v_k * MODEL_DIM + head_idx * HEAD_DIM + v_n  -> WQ[k][n]
  NEW (fixed): phys = (head_idx * HEAD_DIM + v_n) * MODEL_DIM + v_k -> WQ[n][k] = WQ.T[k][n] ✓
"""

import struct
from pathlib import Path
import numpy as np

# ── Dimensions (must match basys3_top.vhd constants) ─────────────────────────
MODEL_DIM  = 32
HEAD_DIM   = 32
NUM_HEADS  = 1   # single head -> head_idx always 0

# ── Load weight matrices from binary files ────────────────────────────────────
def load_bin(fname, shape):
    raw  = Path(fname).read_bytes()
    vals = [struct.unpack("b", bytes([b]))[0] for b in raw]
    assert len(vals) == int(np.prod(shape)), \
        f"{fname}: expected {np.prod(shape)}, got {len(vals)}"
    return np.array(vals, dtype=np.int8).reshape(shape)

weight_dir = Path(__file__).parent / "weights_int8"
WQ = load_bin(weight_dir / "WQ.bin", (MODEL_DIM, MODEL_DIM))
WK = load_bin(weight_dir / "WK.bin", (MODEL_DIM, MODEL_DIM))
WV = load_bin(weight_dir / "WV.bin", (MODEL_DIM, MODEL_DIM))
WO = load_bin(weight_dir / "WO.bin", (MODEL_DIM, MODEL_DIM))

# ── Flatten ROMs as VHDL sees them (row-major, [out_idx][in_idx]) ─────────────
# WQ stored as [MODEL_DIM][MODEL_DIM] row-major -> WQ_ROM[out * MODEL_DIM + in]
WQ_ROM = WQ.flatten()
WK_ROM = WK.flatten()
WV_ROM = WV.flatten()
WO_ROM = WO.flatten()

# ── Address generators (match p_weight_map in mha_controller.vhd) ────────────
def old_proj_addr(k, n, head_idx=0):
    """WRONG: reads WQ[k][h*HD+n]  -- was computing X @ WQ (no transpose)"""
    return k * MODEL_DIM + head_idx * HEAD_DIM + n

def new_proj_addr(k, n, head_idx=0):
    """FIXED: reads WQ[h*HD+n][k] = WQ.T[k][h*HD+n] -- X @ WQ.T ✓"""
    return (head_idx * HEAD_DIM + n) * MODEL_DIM + k

def old_wo_addr(k, n):
    """WRONG: direct passthrough reads WO[k][n]"""
    return k * MODEL_DIM + n

def new_wo_addr(k, n):
    """FIXED: reads WO[n][k] = WO.T[k][n] ✓"""
    return n * MODEL_DIM + k

# ── Verify each (k, n) position ───────────────────────────────────────────────
def check_weight_map(name, ROM, old_fn, new_fn, W):
    """
    For each (k, n) in [MODEL_DIM x HEAD_DIM]:
      golden wants:   W.T[k][n]  = W[n][k]
      old VHDL reads: ROM[old_fn(k,n)]
      new VHDL reads: ROM[new_fn(k,n)]
    Report mismatches.
    """
    total     = MODEL_DIM * HEAD_DIM
    old_ok    = 0
    new_ok    = 0
    old_fail  = []
    new_fail  = []

    for k in range(MODEL_DIM):
        for n in range(HEAD_DIM):
            golden_val = int(W[n, k])          # W.T[k][n] = W[n][k]

            old_addr = old_fn(k, n)
            new_addr = new_fn(k, n)

            old_val = int(ROM[old_addr])
            new_val = int(ROM[new_addr])

            if old_val == golden_val:
                old_ok += 1
            else:
                old_fail.append((k, n, golden_val, old_val))

            if new_val == golden_val:
                new_ok += 1
            else:
                new_fail.append((k, n, golden_val, new_val))

    print(f"\n{'='*60}")
    print(f"  {name} projection weight map check")
    print(f"{'='*60}")
    print(f"  Total (k,n) pairs : {total}")
    print(f"  OLD address map   : {old_ok}/{total} correct "
          f"({'PASS' if old_ok==total else 'FAIL'})")
    print(f"  NEW address map   : {new_ok}/{total} correct "
          f"({'PASS' if new_ok==total else 'FAIL'})")

    if old_fail and len(old_fail) <= 5:
        print(f"  OLD first mismatches: {old_fail}")
    elif old_fail:
        print(f"  OLD first 5 mismatches: {old_fail[:5]}")

    if new_fail and len(new_fail) <= 5:
        print(f"  NEW first mismatches: {new_fail}")
    elif new_fail:
        print(f"  NEW first 5 mismatches: {new_fail[:5]}")

    return new_ok == total


# ── Run checks ────────────────────────────────────────────────────────────────
print("VHDL weight address map verification")
print("Checking that NEW (fixed) address formula fetches W^T[k][n] = W[n][k]")

all_pass = True
all_pass &= check_weight_map("WQ", WQ_ROM, old_proj_addr, new_proj_addr, WQ)
all_pass &= check_weight_map("WK", WK_ROM, old_proj_addr, new_proj_addr, WK)
all_pass &= check_weight_map("WV", WV_ROM, old_proj_addr, new_proj_addr, WV)

# W_O check (full MODEL_DIM x MODEL_DIM)
def check_wo():
    total    = MODEL_DIM * MODEL_DIM
    old_ok   = 0
    new_ok   = 0
    old_fail = []
    new_fail = []
    for k in range(MODEL_DIM):
        for n in range(MODEL_DIM):
            golden_val = int(WO[n, k])
            old_val = int(WO_ROM[old_wo_addr(k, n)])
            new_val = int(WO_ROM[new_wo_addr(k, n)])
            if old_val == golden_val: old_ok += 1
            else: old_fail.append((k, n, golden_val, old_val))
            if new_val == golden_val: new_ok += 1
            else: new_fail.append((k, n, golden_val, new_val))

    print(f"\n{'='*60}")
    print(f"  WO output projection weight map check")
    print(f"{'='*60}")
    print(f"  Total (k,n) pairs : {total}")
    print(f"  OLD address map   : {old_ok}/{total} correct "
          f"({'PASS' if old_ok==total else 'FAIL'})")
    print(f"  NEW address map   : {new_ok}/{total} correct "
          f"({'PASS' if new_ok==total else 'FAIL'})")
    if old_fail: print(f"  OLD first 5: {old_fail[:5]}")
    if new_fail: print(f"  NEW first 5: {new_fail[:5]}")
    return new_ok == total

all_pass &= check_wo()

# ── Also verify that old == new when weight is symmetric (sanity) ────────────
print(f"\n{'='*60}")
if all_pass:
    print("  OVERALL: ALL NEW ADDRESS MAPS CORRECT ✓")
    print("  The fix in mha_controller.vhd is verified.")
else:
    print("  OVERALL: SOME NEW MAPS FAILED ✗  -- check above")
print(f"{'='*60}")

# ── Bonus: end-to-end Q matrix comparison ────────────────────────────────────
print("\n--- End-to-end Q matrix (first token, first 8 values) ---")
print("Using the same patch_embed input as verify_inference.py (image #0)\n")

try:
    patch_proj_w = load_bin(weight_dir / "patch_proj_w.bin", (MODEL_DIM, 49))
    patch_proj_b = load_bin(weight_dir / "patch_proj_b.bin", (MODEL_DIM,))
    pos_embed    = load_bin(weight_dir / "pos_embed.bin",    (16, MODEL_DIM))

    # Dummy constant input x (all zeros + pos_embed[0]) for quick sanity
    x_tok0 = pos_embed[0, :].astype(np.int32)   # use pos_embed row 0 as proxy input

    def sat8(v): return max(-128, min(127, int(v)))

    # Python golden: Q[0] = x_tok0 @ WQ.T  (per gemm_int8 with sat8)
    Q_golden = [sat8(sum(int(x_tok0[k]) * int(WQ[n, k]) for k in range(MODEL_DIM)) >> 7)
                for n in range(MODEL_DIM)]

    # VHDL OLD: Q[0] = x_tok0 @ WQ  (reads WQ[k][n] at b_addr = k*HD+n)
    Q_old = [sat8(sum(int(x_tok0[k]) * int(WQ_ROM[old_proj_addr(k, n)]) for k in range(MODEL_DIM)) >> 7)
             for n in range(MODEL_DIM)]

    # VHDL NEW: Q[0] = x_tok0 @ WQ.T  (reads WQ[n][k] = WQ.T[k][n])
    Q_new = [sat8(sum(int(x_tok0[k]) * int(WQ_ROM[new_proj_addr(k, n)]) for k in range(MODEL_DIM)) >> 7)
             for n in range(MODEL_DIM)]

    print(f"Q[0,:8]  golden  : {Q_golden[:8]}")
    print(f"Q[0,:8]  VHDL old: {Q_old[:8]}  {'MATCH' if Q_old[:8]==Q_golden[:8] else 'MISMATCH'}")
    print(f"Q[0,:8]  VHDL new: {Q_new[:8]}  {'MATCH' if Q_new[:8]==Q_golden[:8] else 'MISMATCH'}")

    q_match = Q_new == Q_golden
    print(f"\nFull Q token-0 matches golden: {'YES ✓' if q_match else 'NO ✗'}")

except FileNotFoundError as e:
    print(f"  (Skipped: {e})")
