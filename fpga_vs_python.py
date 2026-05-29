#!/usr/bin/env python3
"""
fpga_vs_python.py
=================
Sends 20 MNIST images to FPGA via UART and compares predictions
against the Python golden model.

Reports per-image:
  - True label
  - Python prediction
  - FPGA prediction
  - Match (FPGA == Python?)
  - Correct (FPGA == True?)

Usage:
  python fpga_vs_python.py --port COM4
"""

import argparse
import struct
import math
import time
import serial
from pathlib import Path
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port",    default="COM4")
parser.add_argument("--baud",    type=int, default=115200)
parser.add_argument("--count",   type=int, default=20)
parser.add_argument("--timeout", type=float, default=60.0)
args = parser.parse_args()

# ── Model dims ────────────────────────────────────────────────────────────────
PATCH_SIZE  = 7
SEQ_LEN     = 16
D_MODEL     = 32
HEAD_DIM    = 32
D_FF        = 64
Q_SCALE     = 128
LOG_SL      = 4
_LOG_SQRT_HD = 2

def sat8(x): return max(-128, min(127, int(x)))
def transpose(a): return [list(col) for col in zip(*a)]

# GELU LUT
_SQRT_2_PI = 0.7978845608028654
def _build_gelu_lut():
    lut = []
    for i in range(256):
        x_int  = i if i < 128 else i - 256
        x_real = x_int / 128.0
        t      = _SQRT_2_PI * (x_real + 0.044715 * x_real**3)
        y_real = 0.5 * x_real * (1.0 + math.tanh(t))
        lut.append(max(-128, min(127, int(y_real * 128.0))))
    return lut
GELU_LUT = _build_gelu_lut()
def gelu_int8(x): return GELU_LUT[x & 0xFF]

# Softmax LUT
_SM_LUT_DEPTH = 256
_SM_X_MIN     = -10.0
def _build_softmax_lut():
    lut = []
    for i in range(_SM_LUT_DEPTH):
        x_real = _SM_X_MIN + i * (-_SM_X_MIN / (_SM_LUT_DEPTH - 1))
        lut.append(int(math.floor(math.exp(x_real) * (1 << 16) + 0.5)))
    return lut
_EXP_LUT_Q16 = _build_softmax_lut()

def _exp_q7_from_diff(diff):
    if diff >= 0: return 127
    magnitude = min(-diff, Q_SCALE)
    scaled    = (magnitude * (_SM_LUT_DEPTH - 1)) // (10 * Q_SCALE)
    idx       = min(_SM_LUT_DEPTH - 1, (_SM_LUT_DEPTH - 1) - scaled)
    return min(127, _EXP_LUT_Q16[idx] >> 9)

def softmax_int8(row):
    row_max = max(row)
    exps    = [_exp_q7_from_diff(x - row_max) for x in row]
    denom   = sum(exps)
    if denom <= 0: return [Q_SCALE // len(row)] * len(row)
    return [sat8((e * Q_SCALE) // denom) for e in exps]

def gemm_int8(A, B, bias=None):
    M, K, N = len(A), len(A[0]), len(B[0])
    out = []
    for m in range(M):
        row = []
        for n in range(N):
            acc = (bias[n] * Q_SCALE) if bias else 0
            for k in range(K):
                acc += A[m][k] * B[k][n]
            row.append(sat8(acc >> 7))
        out.append(row)
    return out

def _leading_one(x):
    if x == 0: return 0
    for i in range(47, -1, -1):
        if (x >> i) & 1: return i
    return 0

def layernorm_lod_int8(tokens):
    vb = 5; frac = 7; out = []
    for row in tokens:
        s = sum(row); s_sq = sum(x * x for x in row)
        mean = s >> vb; var = max(0, (s_sq >> vb) - mean * mean)
        lod = _leading_one(var); shift = (lod + 1) // 2
        norm_row = []
        for x in row:
            diff = x - mean; net = shift - frac
            norm_row.append(sat8(diff >> net if net >= 0 else diff << (-net)))
        out.append(norm_row)
    return out

def add_sat8(a, b):
    return [[sat8(x + y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

# ── Load weights ──────────────────────────────────────────────────────────────
def load_bin(fname, shape):
    raw  = Path(fname).read_bytes()
    vals = [struct.unpack("b", bytes([b]))[0] for b in raw]
    return np.array(vals, dtype=np.int8).reshape(shape).tolist()

d = Path("./weights_int8")
WQ            = load_bin(d / "WQ.bin",           (D_MODEL, D_MODEL))
WK            = load_bin(d / "WK.bin",           (D_MODEL, D_MODEL))
WV            = load_bin(d / "WV.bin",           (D_MODEL, D_MODEL))
WO            = load_bin(d / "WO.bin",           (D_MODEL, D_MODEL))
W1            = load_bin(d / "W1.bin",           (D_FF, D_MODEL))
b1            = load_bin(d / "b1.bin",           (D_FF,))
W2            = load_bin(d / "W2.bin",           (D_MODEL, D_FF))
b2            = load_bin(d / "b2.bin",           (D_MODEL,))
patch_proj_w  = load_bin(d / "patch_proj_w.bin", (D_MODEL, PATCH_SIZE * PATCH_SIZE))
patch_proj_b  = load_bin(d / "patch_proj_b.bin", (D_MODEL,))
pos_embed     = load_bin(d / "pos_embed.bin",    (SEQ_LEN, D_MODEL))
classifier_w  = load_bin(d / "classifier_w.bin", (10, D_MODEL))
classifier_b  = load_bin(d / "classifier_b.bin", (10,))

# ── MNIST loader ──────────────────────────────────────────────────────────────
def load_mnist(count):
    base = Path("./data/MNIST/raw")
    imgs, lbls = [], []
    with (base / "t10k-images-idx3-ubyte").open("rb") as f:
        f.read(16)
        for _ in range(count):
            imgs.append(list(f.read(28 * 28)))
    with (base / "t10k-labels-idx1-ubyte").open("rb") as f:
        f.read(8)
        for _ in range(count):
            lbls.append(f.read(1)[0])
    return imgs, lbls

def norm_px(p): return sat8(int(round((p / 255.0 - 0.1307) / 0.3081 * 128.0)))

def pix_addr(p, k):
    return (p // 4 * PATCH_SIZE + k // PATCH_SIZE) * 28 + (p % 4 * PATCH_SIZE + k % PATCH_SIZE)

# ── Python golden model ───────────────────────────────────────────────────────
def python_predict(pixels):
    norm = [norm_px(p) for p in pixels]
    x = []
    for p in range(SEQ_LEN):
        row = []
        for d_idx in range(D_MODEL):
            acc = patch_proj_b[d_idx] * Q_SCALE
            for k in range(PATCH_SIZE * PATCH_SIZE):
                acc += norm[pix_addr(p, k)] * patch_proj_w[d_idx][k]
            row.append(sat8(sat8(acc >> 7) + pos_embed[p][d_idx]))
        x.append(row)
    Q = gemm_int8(x, transpose(WQ))
    K = gemm_int8(x, transpose(WK))
    V = gemm_int8(x, transpose(WV))
    scores = [[sat8(sum(Q[i][d] * K[j][d] for d in range(D_MODEL)) >> (7 + _LOG_SQRT_HD))
               for j in range(SEQ_LEN)] for i in range(SEQ_LEN)]
    probs = [softmax_int8(row) for row in scores]
    ctx = [[sat8(sum(probs[i][j] * V[j][d] for j in range(SEQ_LEN)) >> 7)
            for d in range(D_MODEL)] for i in range(SEQ_LEN)]
    mha = gemm_int8(ctx, transpose(WO))
    y1  = layernorm_lod_int8(add_sat8(x, mha))
    fc1 = gemm_int8(y1, transpose(W1), b1)
    act = [[gelu_int8(v) for v in row] for row in fc1]
    ffn = gemm_int8(act, transpose(W2), b2)
    y2  = layernorm_lod_int8(add_sat8(y1, ffn))
    gap = [sat8(sum(y2[t][f] for t in range(SEQ_LEN)) >> LOG_SL) for f in range(D_MODEL)]
    logits = gemm_int8([gap], transpose(classifier_w), classifier_b)[0]
    return int(np.argmax(logits))

# ── FPGA inference via UART ───────────────────────────────────────────────────
def fpga_predict(ser, pixels):
    # Send RAW uint8 pixels (0..255).  The FPGA normalises internally via its
    # NORM_LUT, so sending pre-normalised values here would double-normalise
    # (e.g. background 0 -> norm -54 -> byte 202 -> NORM_LUT -> 127), turning the
    # whole image into a near-constant input -> degenerate output -> always 0.
    raw = bytes([p & 0xFF for p in pixels])
    ser.reset_input_buffer()
    ser.write(raw)
    # Read ACK (0xA5)
    ack = ser.read(1)
    if not ack or ack[0] != 0xA5:
        return None, f"bad ACK: {ack.hex() if ack else 'timeout'}"
    # Read class byte
    res = ser.read(1)
    if not res:
        return None, "timeout waiting for result"
    return res[0], None

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {args.count} MNIST images...")
images, labels = load_mnist(args.count)

print("Computing Python predictions...")
# Use the canonical golden_model (with LN_HEADROOM) as the reference, not the
# local python_predict copy (which lacks the headroom fix and is kept only for
# reference).  This guarantees the comparison uses the SAME math as the VHDL.
import golden_model as _gm
py_preds = [_gm.predict(img) for img in images]

print(f"\nOpening {args.port} at {args.baud} baud...")
ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
time.sleep(0.5)

print("\n" + "="*72)
print(f"{'#':>3}  {'True':>5}  {'Python':>7}  {'FPGA':>5}  {'Py==FPGA':>9}  {'FPGA OK':>7}")
print("="*72)

fpga_correct   = 0
fpga_match_py  = 0
fpga_errors    = 0
results        = []

for idx, (pixels, true_label) in enumerate(zip(images, labels)):
    py_pred = py_preds[idx]
    fp_pred, err = fpga_predict(ser, pixels)

    if err:
        print(f"{idx:>3}  {true_label:>5}  {py_pred:>7}  {'ERR':>5}  {'-':>9}  {'-':>7}  ({err})")
        fpga_errors += 1
        results.append((true_label, py_pred, None))
        continue

    match_py = "✓" if fp_pred == py_pred  else "✗"
    correct  = "✓" if fp_pred == true_label else "✗"
    if fp_pred == true_label: fpga_correct  += 1
    if fp_pred == py_pred:    fpga_match_py += 1

    print(f"{idx:>3}  {true_label:>5}  {py_pred:>7}  {fp_pred:>5}  {match_py:>9}  {correct:>7}")
    results.append((true_label, py_pred, fp_pred))

ser.close()

valid = args.count - fpga_errors
print("="*72)
print(f"FPGA accuracy vs true label : {fpga_correct}/{valid} = {fpga_correct/valid*100:.1f}%")
print(f"FPGA matches Python golden  : {fpga_match_py}/{valid} = {fpga_match_py/valid*100:.1f}%")
print(f"Python accuracy             : {sum(1 for t,p,_ in results if t==p and p is not None)}/{valid} = "
      f"{sum(1 for t,p,_ in results if t==p and p is not None)/valid*100:.1f}%")
if fpga_errors:
    print(f"UART errors                 : {fpga_errors}")

# Distribution of FPGA predictions
if valid > 0:
    from collections import Counter
    fp_vals = [fp for _,_,fp in results if fp is not None]
    dist = Counter(fp_vals)
    print(f"\nFPGA prediction distribution: {dict(sorted(dist.items()))}")
    py_dist = Counter(py_preds)
    print(f"Python prediction distribution: {dict(sorted(py_dist.items()))}")
