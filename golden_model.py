#!/usr/bin/env python3
"""
golden_model.py  --  Bit-exact Python reference for the MNIST ViT
=================================================================
Importable module wrapping the integer golden model.  This is the SAME
arithmetic the VHDL pipeline implements (verified bit-exact, 0/512 mismatches
per stage in GHDL simulation of basys3_top).

Usage:
    import golden_model as gm
    pred = gm.predict(raw_pixels)          # raw_pixels: list[int] 0..255, len 784

The module loads the int8 weights from ./weights_int8 lazily on first call,
so importing it is cheap and side-effect free.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import List

# ── Model dims (must match basys3_top.vhd) ──────────────────────────────────
PATCH_SIZE   = 7
SEQ_LEN      = 16
D_MODEL      = 32
HEAD_DIM     = 32
D_FF         = 64
Q_SCALE      = 128
LOG_SL       = 4
_LOG_SQRT_HD = 2          # >>2 ~ /sqrt(32)

_WEIGHT_DIR = Path(__file__).resolve().parent / "weights_int8"


def sat8(x: int) -> int:
    return max(-128, min(127, int(x)))


def _transpose(a):
    return [list(col) for col in zip(*a)]


# ── GELU LUT (truncate toward zero — matches psum_activation.vhd) ───────────
_SQRT_2_PI = 0.7978845608028654


def _build_gelu_lut() -> List[int]:
    lut = []
    for i in range(256):
        x_int  = i if i < 128 else i - 256
        x_real = x_int / 128.0
        t      = _SQRT_2_PI * (x_real + 0.044715 * x_real ** 3)
        y_real = 0.5 * x_real * (1.0 + math.tanh(t))
        lut.append(max(-128, min(127, int(y_real * 128.0))))  # int() truncates
    return lut


_GELU_LUT = _build_gelu_lut()


def _gelu_int8(x: int) -> int:
    return _GELU_LUT[x & 0xFF]


# ── Softmax LUT (Q16 -> Q7 via >>9, matches softmax.vhd) ─────────────────────
_SM_LUT_DEPTH = 256
_SM_X_MIN     = -10.0


def _build_softmax_lut() -> List[int]:
    lut = []
    for i in range(_SM_LUT_DEPTH):
        x_real = _SM_X_MIN + i * (-_SM_X_MIN / (_SM_LUT_DEPTH - 1))
        lut.append(int(math.floor(math.exp(x_real) * (1 << 16) + 0.5)))
    return lut


_EXP_LUT_Q16 = _build_softmax_lut()


def _exp_q7_from_diff(diff: int) -> int:
    if diff >= 0:
        return 127
    magnitude = min(-diff, Q_SCALE)
    scaled    = (magnitude * (_SM_LUT_DEPTH - 1)) // (10 * Q_SCALE)
    idx       = min(_SM_LUT_DEPTH - 1, (_SM_LUT_DEPTH - 1) - scaled)
    return min(127, _EXP_LUT_Q16[idx] >> 9)


def _softmax_int8(row):
    row_max = max(row)
    exps    = [_exp_q7_from_diff(x - row_max) for x in row]
    denom   = sum(exps)
    if denom <= 0:
        return [Q_SCALE // len(row)] * len(row)
    return [sat8((e * Q_SCALE) // denom) for e in exps]


def _gemm_int8(A, B, bias=None):
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


def _leading_one(x: int) -> int:
    if x == 0:
        return 0
    for i in range(47, -1, -1):
        if (x >> i) & 1:
            return i
    return 0


# LN_HEADROOM: extra right-shift on the LayerNorm output so standardized values
# (std~1, up to +-3) fit in Q1.7 instead of being hard-clipped at +-1.0.
# frac = 7 - LN_HEADROOM.  Must match layernorm.vhd and qat_hw_exact.py.
LN_HEADROOM = 2


def _layernorm_lod_int8(tokens):
    vb, frac = 5, 7 - LN_HEADROOM
    out = []
    for row in tokens:
        s = sum(row)
        s_sq = sum(x * x for x in row)
        mean = s >> vb
        var = max(0, (s_sq >> vb) - mean * mean)
        lod = _leading_one(var)
        shift = (lod + 1) // 2
        norm_row = []
        for x in row:
            diff = x - mean
            net = shift - frac
            norm_row.append(sat8(diff >> net if net >= 0 else diff << (-net)))
        out.append(norm_row)
    return out


def _add_sat8(a, b):
    return [[sat8(x + y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _norm_px(p: int) -> int:
    return sat8(int(round((p / 255.0 - 0.1307) / 0.3081 * 128.0)))


def _pix_addr(p: int, k: int) -> int:
    return (p // 4 * PATCH_SIZE + k // PATCH_SIZE) * 28 + (p % 4 * PATCH_SIZE + k % PATCH_SIZE)


# ── Lazy weight loading ──────────────────────────────────────────────────────
_W = None


def _load_bin(fname, shape):
    raw = Path(fname).read_bytes()
    vals = [struct.unpack("b", bytes([b]))[0] for b in raw]
    n = 1
    for s in shape:
        n *= s
    assert len(vals) == n, f"{fname}: expected {n}, got {len(vals)}"
    # reshape into nested lists
    if len(shape) == 1:
        return vals
    rows, cols = shape
    return [vals[r * cols:(r + 1) * cols] for r in range(rows)]


def _weights():
    global _W
    if _W is None:
        d = _WEIGHT_DIR
        _W = dict(
            WQ=_load_bin(d / "WQ.bin", (D_MODEL, D_MODEL)),
            WK=_load_bin(d / "WK.bin", (D_MODEL, D_MODEL)),
            WV=_load_bin(d / "WV.bin", (D_MODEL, D_MODEL)),
            WO=_load_bin(d / "WO.bin", (D_MODEL, D_MODEL)),
            W1=_load_bin(d / "W1.bin", (D_FF, D_MODEL)),
            b1=_load_bin(d / "b1.bin", (D_FF,)),
            W2=_load_bin(d / "W2.bin", (D_MODEL, D_FF)),
            b2=_load_bin(d / "b2.bin", (D_MODEL,)),
            patch_proj_w=_load_bin(d / "patch_proj_w.bin", (D_MODEL, PATCH_SIZE * PATCH_SIZE)),
            patch_proj_b=_load_bin(d / "patch_proj_b.bin", (D_MODEL,)),
            pos_embed=_load_bin(d / "pos_embed.bin", (SEQ_LEN, D_MODEL)),
            classifier_w=_load_bin(d / "classifier_w.bin", (10, D_MODEL)),
            classifier_b=_load_bin(d / "classifier_b.bin", (10,)),
        )
    return _W


def predict(pixels) -> int:
    """Run the bit-exact integer ViT on 784 raw uint8 pixels, return class 0-9."""
    W = _weights()
    norm = [_norm_px(p) for p in pixels]
    x = []
    for p in range(SEQ_LEN):
        row = []
        for d_idx in range(D_MODEL):
            acc = W["patch_proj_b"][d_idx] * Q_SCALE
            for k in range(PATCH_SIZE * PATCH_SIZE):
                acc += norm[_pix_addr(p, k)] * W["patch_proj_w"][d_idx][k]
            row.append(sat8(sat8(acc >> 7) + W["pos_embed"][p][d_idx]))
        x.append(row)
    Q = _gemm_int8(x, _transpose(W["WQ"]))
    K = _gemm_int8(x, _transpose(W["WK"]))
    V = _gemm_int8(x, _transpose(W["WV"]))
    scores = [[sat8(sum(Q[i][d] * K[j][d] for d in range(D_MODEL)) >> (7 + _LOG_SQRT_HD))
               for j in range(SEQ_LEN)] for i in range(SEQ_LEN)]
    probs = [_softmax_int8(r) for r in scores]
    ctx = [[sat8(sum(probs[i][j] * V[j][d] for j in range(SEQ_LEN)) >> 7)
            for d in range(D_MODEL)] for i in range(SEQ_LEN)]
    mha = _gemm_int8(ctx, _transpose(W["WO"]))
    y1 = _layernorm_lod_int8(_add_sat8(x, mha))
    fc1 = _gemm_int8(y1, _transpose(W["W1"]), W["b1"])
    act = [[_gelu_int8(v) for v in row] for row in fc1]
    ffn = _gemm_int8(act, _transpose(W["W2"]), W["b2"])
    y2 = _layernorm_lod_int8(_add_sat8(y1, ffn))
    gap = [sat8(sum(y2[t][f] for t in range(SEQ_LEN)) >> LOG_SL) for f in range(D_MODEL)]
    logits = _gemm_int8([gap], _transpose(W["classifier_w"]), W["classifier_b"])[0]
    # argmax (first-max wins, matches classifier.vhd strict-greater argmax)
    best_i, best_v = 0, logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_v:
            best_v, best_i = logits[i], i
    return best_i


if __name__ == "__main__":
    # quick self-test on first 20 MNIST test images
    base = Path(__file__).resolve().parent / "data" / "MNIST" / "raw"
    with (base / "t10k-images-idx3-ubyte").open("rb") as f:
        f.read(16)
        imgs = [list(f.read(784)) for _ in range(20)]
    with (base / "t10k-labels-idx1-ubyte").open("rb") as f:
        f.read(8)
        lbls = [f.read(1)[0] for _ in range(20)]
    correct = 0
    for i, (im, lb) in enumerate(zip(imgs, lbls)):
        p = predict(im)
        ok = p == lb
        correct += ok
        print(f"#{i:2d} true={lb} pred={p} {'OK' if ok else 'X'}")
    print(f"accuracy {correct}/20 = {correct/20*100:.1f}%")
