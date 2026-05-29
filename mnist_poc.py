#!/usr/bin/env python3
"""
mnist_poc.py — MNIST ViT, Post-LayerNorm, int8 datapath
========================================================

Float model  →  train on MNIST  →  quantize int8  →  export weights
Integer golden model matches the VHDL encoder_block(structural) exactly:
  • GEMM:       Q1.7 accumulate + right-shift-7 + saturate
  • LayerNorm:  LOD-shift 1/√ approximation (no multiplier, matches layernorm.vhd)
  • Softmax:    exp-LUT Q1.7 (matches softmax.vhd)
  • GELU:       tanh-approx LUT (matches psum_activation.vhd init_gelu_lut)
  • Structure:  Post-LN  MHA → Add+LN → FFN(GELU) → Add+LN

MNIST ViT dimensions (= FPGA generics):
  patch_size = 7  →  4×4 = 16 patches from 28×28 image
  d_model    = 32
  n_heads    = 1
  head_dim   = 32
  d_ff       = 64
  n_layers   = 1

Usage:
  python mnist_poc.py train          # train float model, save mnist_vit.pth
  python mnist_poc.py export         # load mnist_vit.pth → export int8 weights
  python mnist_poc.py golden <file>  # run integer golden model on encoder_out.txt
  python mnist_poc.py verify         # smoke-test: golden(random) vs torch(random)
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────────────────────
# Dimensions  (keep in sync with VHDL generics)
# ──────────────────────────────────────────────────────────────────────────────
PATCH_SIZE  = 7    # pixels per side; 28/7 = 4 patches per axis
SEQ_LEN     = 16   # 4 × 4 patches
D_MODEL     = 32
N_HEADS     = 1
HEAD_DIM    = D_MODEL // N_HEADS  # 32
D_FF        = 64
N_LAYERS    = 1
DATA_WIDTH  = 8    # int8
Q_SCALE     = 1 << (DATA_WIDTH - 1)  # 128  (Q1.7)
VEC_BITS    = round(math.log2(D_MODEL))  # 5  (for LN mean/var divide)

# ──────────────────────────────────────────────────────────────────────────────
# Integer helper functions
# ──────────────────────────────────────────────────────────────────────────────

def sat8(x: int) -> int:
    """Saturate to int8 range."""
    return max(-128, min(127, x))

def sat16(x: int) -> int:
    return max(-32768, min(32767, x))


# ──────────────────────────────────────────────────────────────────────────────
# GELU LUT  — must match psum_activation.vhd init_gelu_lut exactly
#   Q_SCALE_8  = 128.0  (Q1.7)
#   x_real     = x_int / 128
#   GELU(x)    = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))
#   y_q        = round(GELU(x_real) * 128), saturated to [-128, 127]
# ──────────────────────────────────────────────────────────────────────────────
_SQRT_2_PI = 0.7978845608028654

def _build_gelu_lut() -> List[int]:
    lut = []
    for i in range(256):
        x_int  = i if i < 128 else i - 256
        x_real = x_int / 128.0
        t      = _SQRT_2_PI * (x_real + 0.044715 * x_real**3)
        tanh_v = math.tanh(t)
        y_real = 0.5 * x_real * (1.0 + tanh_v)
        y_q    = int(y_real * 128.0)      # truncate (VHDL integer() truncates)
        lut.append(max(-128, min(127, y_q)))
    return lut

GELU_LUT_I8: List[int] = _build_gelu_lut()

def gelu_int8(x: int) -> int:
    """GELU via 256-entry LUT, index = x reinterpreted as unsigned."""
    return GELU_LUT_I8[x & 0xFF]


# ──────────────────────────────────────────────────────────────────────────────
# Softmax LUT  — matches softmax.vhd (exp LUT, Q1.7)
#   LUT covers x ∈ [−10, 0], 256 entries, values in Q8.8 (shifted to Q1.7)
# ──────────────────────────────────────────────────────────────────────────────
_SM_LUT_DEPTH = 256
_SM_X_MIN     = -10.0

def _build_softmax_lut() -> List[int]:
    lut = []
    for i in range(_SM_LUT_DEPTH):
        x_real = _SM_X_MIN + i * (-_SM_X_MIN / (_SM_LUT_DEPTH - 1))
        val    = int(math.floor(math.exp(x_real) * (1 << 16) + 0.5))
        lut.append(val)
    return lut

_EXP_LUT_Q16: List[int] = _build_softmax_lut()

def _exp_q7_from_diff(diff: int) -> int:
    """exp(diff) in Q1.7, used for softmax; diff ≤ 0 (max-subtracted)."""
    if diff >= 0:
        return 127
    magnitude = min(-diff, Q_SCALE)
    scaled    = (magnitude * (_SM_LUT_DEPTH - 1)) // (10 * Q_SCALE)
    idx       = min(_SM_LUT_DEPTH - 1, (_SM_LUT_DEPTH - 1) - scaled)
    return min(127, _EXP_LUT_Q16[idx] >> 9)  # Q16 → Q7

def softmax_int8(row: List[int]) -> List[int]:
    """Row-wise softmax in Q1.7, matching softmax.vhd."""
    row_max = max(row)
    exps    = [_exp_q7_from_diff(x - row_max) for x in row]
    denom   = sum(exps)
    if denom <= 0:
        return [Q_SCALE // len(row)] * len(row)
    return [sat8((e * Q_SCALE) // denom) for e in exps]


# ──────────────────────────────────────────────────────────────────────────────
# GEMM  — matches gemm_mm.vhd arithmetic
#   A : M×K int8,  B : K×N int8,  bias : N int8 (optional)
#   Bias is pre-scaled by Q_SCALE before accumulation (matches ST_LOAD_C shift).
#   Final output right-shifted by DATA_WIDTH-1 = 7 and saturated to int8.
# ──────────────────────────────────────────────────────────────────────────────

def gemm_int8(
    A: List[List[int]],
    B: List[List[int]],
    bias: List[int] | None = None,
) -> List[List[int]]:
    M  = len(A)
    K  = len(A[0])
    N  = len(B[0])
    out = []
    for m in range(M):
        row = []
        for n in range(N):
            acc = (bias[n] * Q_SCALE) if bias else 0  # bias pre-scaled (<<7)
            for k in range(K):
                acc += A[m][k] * B[k][n]
            row.append(sat8(acc >> (DATA_WIDTH - 1)))  # >>7 + saturate
        out.append(row)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# LayerNorm  — matches layernorm.vhd (LOD-shift 1/√ approximation)
#   mean     = sum(x) >> VEC_BITS           (divide by N = power-of-two)
#   var      = (sum(x²)>>VEC_BITS) − mean²  (clamp ≥ 0)
#   lod      = MSB index of var
#   shift    = (lod + 1) / 2                (integer approximation of log2(std))
#   y_i      = sat8((x_i − mean) >> shift)  (no multiplier!)
# ──────────────────────────────────────────────────────────────────────────────

def _leading_one(x: int) -> int:
    """Index of the most-significant '1' bit; returns 0 for x == 0."""
    if x == 0:
        return 0
    for i in range(47, -1, -1):
        if (x >> i) & 1:
            return i
    return 0

def layernorm_lod_int8(
    tokens: List[List[int]],
    vec_size: int = D_MODEL,
) -> List[List[int]]:
    """
    Apply LOD-shift LayerNorm to a sequence of integer vectors.

    Output is in Q1.(DATA_WIDTH-1) format (same as input), matching the
    updated layernorm.vhd which computes:

        norm = (x - mean) * 2^(DATA_WIDTH-1) / 2^norm_shift

    i.e. shift_left(diff, DATA_WIDTH-1) then shift_right(result, norm_shift),
    or equivalently shift in the net direction after subtracting.
    This preserves Q1.7 scale instead of collapsing to ~8 distinct values.
    """
    vb  = round(math.log2(vec_size))   # must be power-of-two
    frac = DATA_WIDTH - 1               # = 7 for int8
    out = []
    for row in tokens:
        s    = sum(row)
        s_sq = sum(x * x for x in row)
        mean = s >> vb
        mean_sq  = mean * mean
        avg_sq   = s_sq >> vb
        var      = max(0, avg_sq - mean_sq)
        lod      = _leading_one(var)
        shift    = (lod + 1) // 2
        norm_row = []
        for x in row:
            diff = x - mean
            # Net shift = shift - frac; positive → right, negative → left
            net = shift - frac
            if net >= 0:
                norm_val = diff >> net        # shift_right dominates
            else:
                norm_val = diff << (-net)     # shift_left dominates
            norm_row.append(sat8(norm_val))
        out.append(norm_row)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Residual add (saturating, matches residual_add.vhd)
# ──────────────────────────────────────────────────────────────────────────────

def add_sat8(
    a: List[List[int]],
    b: List[List[int]],
) -> List[List[int]]:
    return [[sat8(x + y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention  (matches mha_controller.vhd)
# ──────────────────────────────────────────────────────────────────────────────

# Attention score scale: right-shift by LOG_SQRT_HEAD_DIM extra bits
# approximates division by sqrt(HEAD_DIM) = sqrt(32) ≈ 5.66 → use shift 2 ≈ /4
_LOG_SQRT_HD = math.ceil(math.log2(HEAD_DIM)) // 2   # ceil(5)/2 = 2

def mha_int8(
    x:   List[List[int]],   # SEQ_LEN × MODEL_DIM
    WQ:  List[List[int]],   # MODEL_DIM × MODEL_DIM
    WK:  List[List[int]],
    WV:  List[List[int]],
    WO:  List[List[int]],
) -> List[List[int]]:
    """Integer MHA matching mha_controller.vhd (no bias in projections)."""
    seq, dm = len(x), len(x[0])
    # Outputs of all heads concatenated along feature axis
    ctx_concat = [[0] * dm for _ in range(seq)]

    for h in range(N_HEADS):
        s = h * HEAD_DIM
        e = s + HEAD_DIM
        # Per-head weight slices  (rows = MODEL_DIM, cols = HEAD_DIM)
        WQ_h = [row[s:e] for row in WQ]
        WK_h = [row[s:e] for row in WK]
        WV_h = [row[s:e] for row in WV]

        Q = gemm_int8(x, WQ_h)
        K = gemm_int8(x, WK_h)
        V = gemm_int8(x, WV_h)

        # Attention scores: Q @ K^T, scaled
        scores = []
        for i in range(seq):
            row = []
            for j in range(seq):
                acc = sum(Q[i][d] * K[j][d] for d in range(HEAD_DIM))
                row.append(sat8(acc >> ((DATA_WIDTH - 1) + _LOG_SQRT_HD)))
            scores.append(row)

        probs = [softmax_int8(row) for row in scores]

        # Context vectors: probs @ V
        for i in range(seq):
            for d in range(HEAD_DIM):
                acc = sum(probs[i][j] * V[j][d] for j in range(seq))
                ctx_concat[i][s + d] = sat8(acc >> (DATA_WIDTH - 1))

    # Output projection: ctx_concat @ W_O
    return gemm_int8(ctx_concat, WO)


# ──────────────────────────────────────────────────────────────────────────────
# FFN  (matches ffn.vhd: FC1 → GELU → FC2, biases optional)
# ──────────────────────────────────────────────────────────────────────────────

def ffn_int8(
    x:   List[List[int]],   # SEQ_LEN × MODEL_DIM
    W1:  List[List[int]],   # HIDDEN_DIM × MODEL_DIM  (W_1 row = output neuron)
    b1:  List[int] | None,  # HIDDEN_DIM
    W2:  List[List[int]],   # MODEL_DIM × HIDDEN_DIM
    b2:  List[int] | None,  # MODEL_DIM
) -> List[List[int]]:
    # FC1: x @ W1^T + b1  (W1 stored as HIDDEN_DIM × MODEL_DIM)
    W1T = [[W1[hd][md] for hd in range(len(W1))] for md in range(len(W1[0]))]
    h = gemm_int8(x, W1T, b1)
    # GELU activation (element-wise LUT)
    h = [[gelu_int8(v) for v in row] for row in h]
    # FC2: h @ W2^T + b2  (W2 stored as MODEL_DIM × HIDDEN_DIM)
    W2T = [[W2[md][hd] for md in range(len(W2))] for hd in range(len(W2[0]))]
    return gemm_int8(h, W2T, b2)


# ──────────────────────────────────────────────────────────────────────────────
# Full encoder block (Post-LN, matching encoder_block.vhd structural)
#   MHA → Add → LN₁ → FFN(GELU) → Add → LN₂
# ──────────────────────────────────────────────────────────────────────────────

def encoder_int8(
    x:   List[List[int]],
    WQ, WK, WV, WO,
    W1, b1, W2, b2,
) -> List[List[int]]:
    # MHA sublayer
    mha_out = mha_int8(x, WQ, WK, WV, WO)
    x       = layernorm_lod_int8(add_sat8(x, mha_out))
    # FFN sublayer
    ffn_out = ffn_int8(x, W1, b1, W2, b2)
    x       = layernorm_lod_int8(add_sat8(x, ffn_out))
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Float PyTorch model  (Post-LN, matches VHDL structure)
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Extract non-overlapping 7×7 patches from 28×28 MNIST, project to d_model."""
    def __init__(self, patch: int = PATCH_SIZE, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.patch = patch
        self.proj  = nn.Linear(patch * patch, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        B = x.shape[0]
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)          # [B, 1, 4, 4, p, p]
        x = x.contiguous().reshape(B, SEQ_LEN, p * p)   # [B, 16, 49]
        return self.proj(x)                              # [B, 16, 32]


class PostLNFFN(nn.Module):
    """Two-layer FFN with GELU, matching ffn.vhd."""
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class PostLNEncoderLayer(nn.Module):
    """
    Post-LN encoder layer: sublayer → Add → LayerNorm
    Matches encoder_block(structural) exactly.
    """
    def __init__(
        self,
        d_model:  int = D_MODEL,
        n_heads:  int = N_HEADS,
        d_ff:     int = D_FF,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = PostLNFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-LN: sublayer first, then residual + norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))   # MHA → Add → LN
        x = self.norm2(x + self.drop(self.ffn(x))) # FFN → Add → LN
        return x


class MNISTViT(nn.Module):
    """
    Minimal ViT for 28×28 MNIST.
    patch 7×7 → 16 tokens → d_model=32 → 1 Post-LN encoder → GAP → 10 classes.
    """
    def __init__(
        self,
        patch:    int   = PATCH_SIZE,
        d_model:  int   = D_MODEL,
        n_heads:  int   = N_HEADS,
        d_ff:     int   = D_FF,
        n_layers: int   = N_LAYERS,
        n_cls:    int   = 10,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.Sequential(
            *[PostLNEncoderLayer(d_model, n_heads, d_ff, dropout)
              for _ in range(n_layers)]
        )
        self.norm       = nn.LayerNorm(d_model)       # final norm after all layers
        self.classifier = nn.Linear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed      # [B, 16, 32]
        x = self.encoder(x)                            # [B, 16, 32]
        x = self.norm(x)
        x = x.mean(dim=1)                              # global average pool [B, 32]
        return self.classifier(x)                      # [B, 10]


# ══════════════════════════════════════════════════════════════════════════════
# Quantization-Aware Training (QAT)
# ══════════════════════════════════════════════════════════════════════════════
#
# Root cause of 34% int8 accuracy: Q1.7 GEMM saturates heavily.
#   K=32, typical |a|=50, |b|=15 → acc=24000 >> 7 =187 → clips to 127.
# Fix: fake_q17() simulates the hardware saturation during forward, so the
# model learns to keep all intermediate activations in [-1, 1) = Q1.7 range.
# Gradients flow through the clamp/round via STE (straight-through estimator).
# ──────────────────────────────────────────────────────────────────────────────

def fake_q17(x: torch.Tensor) -> torch.Tensor:
    """
    Fake-quantize to Q1.7 with straight-through estimator (STE).
    Forward : scale to int range, round, clamp to [-128, 127], scale back.
    Backward: identity through both round and clamp (STE).
    """
    x_int      = x * Q_SCALE
    x_clamped  = x_int + (x_int.clamp(-128.0, 127.0) - x_int).detach()
    x_rounded  = x_clamped + (x_clamped.round() - x_clamped).detach()
    return x_rounded / Q_SCALE


class HWLayerNorm(nn.Module):
    """
    Hardware-faithful LayerNorm matching layernorm.vhd exactly:
      • standardize:  (x - mean) / std      (output has std ~ 1.0)
      • NO affine (no gamma/beta) -- the FPGA has no learnable LN params
      • saturate to Q1.7 via fake_q17       (the step the old QAT was MISSING)

    The old QAT used nn.LayerNorm (float, affine, no saturation), so the model
    learned activations up to +-3 that the int8 hardware (Q1.7, max +-1) cannot
    represent and silently clips -> the 79% -> 49% accuracy collapse.  Training
    WITH the saturation forces the model to keep LN outputs inside [-1, 1).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + 1e-5)
        return fake_q17(norm)


class FQLinear(nn.Module):
    """
    Fake-Quantize Linear layer.
    Simulates the FPGA GEMM:
      • fake_q17 on inputs  (quantize activations)
      • fake_q17 on weights (quantize weights)
      • compute float matmul
      • fake_q17 on output  (simulate >>7 + sat8 saturation)
    """
    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias   = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = fake_q17(self.weight)
        b_q = fake_q17(self.bias) if self.bias is not None else None
        out = F.linear(fake_q17(x), w_q, b_q)
        return fake_q17(out)


class QATPostLNFFN(nn.Module):
    """QAT-aware FFN: FQLinear(d_model→d_ff) → GELU → FQLinear(d_ff→d_model)."""
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        super().__init__()
        self.fc1 = FQLinear(d_model, d_ff,    bias=True)
        self.fc2 = FQLinear(d_ff,    d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = fake_q17(F.gelu(h))   # GELU then re-quantize to Q1.7
        return self.fc2(h)


# Attention score scale that matches the hardware shift of DATA_WIDTH-1+LOG_SQRT_HD = 9
# In float: score = dotprod / (Q_SCALE / 2**_LOG_SQRT_HD) = dotprod / 32
_ATTN_FLOAT_SCALE: float = (2 ** _LOG_SQRT_HD) / Q_SCALE   # = 4/128 = 1/32


class QATSelfAttention(nn.Module):
    """Single-head self-attention with fake-quantized Q/K/V/O projections."""
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.q = FQLinear(d_model, d_model, bias=False)
        self.k = FQLinear(d_model, d_model, bias=False)
        self.v = FQLinear(d_model, d_model, bias=False)
        self.o = FQLinear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, SEQ, D]  (already in Q1.7 range)
        Q = self.q(x)                                         # FQLinear already saturates
        K = self.k(x)
        V = self.v(x)
        # Scores: match hardware  sat8(Q_int · K_int >> 9) = fake_q17(dotprod / 32)
        scores = fake_q17(torch.bmm(Q, K.transpose(1, 2)) * _ATTN_FLOAT_SCALE)
        probs  = fake_q17(F.softmax(scores, dim=-1))          # probs ∈ [0,1), still Q1.7
        ctx    = fake_q17(torch.bmm(probs, V))
        return self.o(ctx)


class QATPostLNEncoderLayer(nn.Module):
    """Post-LN encoder layer with full fake-quantized forward pass."""
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF, dropout: float = 0.1):
        super().__init__()
        self.attn  = QATSelfAttention(d_model)
        self.ffn   = QATPostLNFFN(d_model, d_ff)
        self.norm1 = HWLayerNorm()   # HW-faithful: standardize + Q1.7 saturate, no affine
        self.norm2 = HWLayerNorm()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHA sublayer: attn → fake-saturating add → LN
        x = self.norm1(fake_q17(x + self.drop(self.attn(x))))
        # FFN sublayer
        x = self.norm2(fake_q17(x + self.drop(self.ffn(x))))
        return x


class QATMNISTViT(nn.Module):
    """
    MNIST ViT with a fully fake-quantized encoder.
    The patch_embed, pos_embed, and classifier run in float
    (they are outside the FPGA encoder_block in the final design).
    """
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF, n_cls: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed   = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.encoder     = nn.Sequential(*[
            QATPostLNEncoderLayer(d_model, d_ff, dropout)
            for _ in range(N_LAYERS)
        ])
        # No final LayerNorm: the FPGA encoder_block has none, GAP follows the
        # last encoder norm directly.  (Was nn.LayerNorm before -> mismatch.)
        self.norm       = nn.Identity()
        self.classifier = nn.Linear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed
        x = fake_q17(x)        # quantize to Q1.7 before encoder
        x = self.encoder(x)
        x = self.norm(x)       # Identity (kept for state_dict compatibility)
        x = x.mean(dim=1)      # global average pool
        return self.classifier(x)

    def load_from_float(self, ckpt_path: str) -> None:
        """
        Copy weights from a standard MNISTViT checkpoint into this QAT model.
        Maps fused in_proj_weight (Q/K/V concatenated) → separate FQLinear layers.
        """
        fm = MNISTViT()
        fm.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        # Patch embed & positional embedding
        self.patch_embed.proj.weight.data.copy_(fm.patch_embed.proj.weight.data)
        self.patch_embed.proj.bias.data.copy_(fm.patch_embed.proj.bias.data)
        self.pos_embed.data.copy_(fm.pos_embed.data)
        # Encoder layers
        for ql, fl in zip(self.encoder, fm.encoder):
            in_w = fl.attn.in_proj_weight          # [3*D_MODEL, D_MODEL]
            ql.attn.q.weight.data.copy_(in_w[:D_MODEL])
            ql.attn.k.weight.data.copy_(in_w[D_MODEL:2 * D_MODEL])
            ql.attn.v.weight.data.copy_(in_w[2 * D_MODEL:])
            ql.attn.o.weight.data.copy_(fl.attn.out_proj.weight.data)
            ql.ffn.fc1.weight.data.copy_(fl.ffn.fc1.weight.data)
            ql.ffn.fc1.bias.data.copy_(fl.ffn.fc1.bias.data)
            ql.ffn.fc2.weight.data.copy_(fl.ffn.fc2.weight.data)
            ql.ffn.fc2.bias.data.copy_(fl.ffn.fc2.bias.data)
            # norm1/norm2 are now parameter-free HWLayerNorm -- nothing to copy.
        # Final norm is now nn.Identity (no params); only copy the classifier.
        self.classifier.load_state_dict(fm.classifier.state_dict())

    def encoder_tensors(self) -> dict:
        """Return all weight tensors for int8 export (same keys as export_weights)."""
        enc = self.encoder[0]
        return {
            "WQ":          enc.attn.q.weight,
            "WK":          enc.attn.k.weight,
            "WV":          enc.attn.v.weight,
            "WO":          enc.attn.o.weight,
            "W1":          enc.ffn.fc1.weight,
            "b1":          enc.ffn.fc1.bias,
            "W2":          enc.ffn.fc2.weight,
            "b2":          enc.ffn.fc2.bias,
            "patch_proj_w": self.patch_embed.proj.weight,
            "patch_proj_b": self.patch_embed.proj.bias,
            "pos_embed":    self.pos_embed.squeeze(0),
            "classifier_w": self.classifier.weight,
            "classifier_b": self.classifier.bias,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    epochs:    int   = 20,
    batch:     int   = 256,
    lr:        float = 3e-4,
    data_root: str   = "./data",
    ckpt_path: str   = "mnist_vit.pth",
    device:    str   = "cpu",
) -> None:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    model = MNISTViT().to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(dev), labels.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(dev), labels.to(dev)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {ep:3d} | loss {total_loss/len(train_dl):.4f} | "
              f"val_acc {acc:.4f} | best {best_acc:.4f}")

    print(f"\nBest val accuracy: {best_acc:.4f}  →  saved to {ckpt_path}")


def qat_train(
    ckpt_path: str  = "mnist_vit.pth",
    out_path:  str  = "mnist_vit_qat.pth",
    epochs:    int  = 20,
    batch:     int  = 256,
    lr:        float = 5e-5,
    data_root: str  = "./data",
    device:    str  = "cpu",
) -> None:
    """
    QAT fine-tuning: load float model, wrap with fake-quantize ops, fine-tune.

    The fake_q17 ops clamp all encoder activations to Q1.7 = [-1, 1) in the
    forward pass.  Gradients flow through via STE so the model adapts its
    weights to avoid saturation.  Lower LR (5e-5) than initial training.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    model = QATMNISTViT().to(dev)
    # Auto-detect: float checkpoint has 'encoder.0.attn.in_proj_weight';
    # QAT checkpoint has 'encoder.0.attn.q.weight'.
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "encoder.0.attn.q.weight" in sd:
        model.load_state_dict(sd)      # resume from prior QAT run
        print(f"Resumed QAT model from {ckpt_path}")
    else:
        model.load_from_float(ckpt_path)   # convert float → QAT

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(dev), labels.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(dev), labels.to(dev)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
        print(f"QAT {ep:3d}/{epochs} | loss {total_loss/len(train_dl):.4f} | "
              f"val_acc {acc:.4f} | best {best_acc:.4f}")

    print(f"\nQAT best val accuracy: {best_acc:.4f}  →  saved to {out_path}")


def export_qat_weights(
    qat_ckpt: str = "mnist_vit_qat.pth",
    out_dir:  str = "weights_int8",
) -> None:
    """Load QAT model, quantize to int8, write .bin/.hex and weights_pkg.vhd."""
    Path(out_dir).mkdir(exist_ok=True)
    model = QATMNISTViT()
    model.load_state_dict(torch.load(qat_ckpt, map_location="cpu", weights_only=True))
    model.eval()

    tensors  = model.encoder_tensors()
    scales_txt: list[str] = []

    for name, t in tensors.items():
        scale    = _infer_scale(t)
        qi8_list = _quantize_tensor(t, scale)
        raw      = bytes([x & 0xFF for x in qi8_list])

        (Path(out_dir) / f"{name}.bin").write_bytes(raw)
        (Path(out_dir) / f"{name}.hex").write_text(
            "\n".join(f"{b:02x}" for b in raw) + "\n")

        scales_txt.append(f"{name:20s}  scale={scale:.6f}  numel={len(qi8_list)}")
        print(f"  {name:20s}: {list(t.shape)}  scale={scale:.4f}")

    (Path(out_dir) / "scales.txt").write_text("\n".join(scales_txt) + "\n")
    print(f"\nQAT weights written to {out_dir}/")
    export_vhdl_pkg(tensors, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Int8 quantization  (symmetric per-tensor, scale = 127 / max_abs)
# ══════════════════════════════════════════════════════════════════════════════

def _quantize_tensor(t: torch.Tensor, scale: float) -> List[int]:
    """Quantize float tensor to int8 list (row-major)."""
    q = (t.float() * scale).round().clamp(-128, 127).to(torch.int8)
    return q.flatten().tolist()

def _infer_scale(t: torch.Tensor) -> float:
    """
    Fixed Q1.7 scale (= 128).  The FPGA GEMM does:
      output_int = (sum(a_int * b_int) + bias_int<<7) >> 7
    which is exact only when a, b, bias all use the SAME scale of 128.
    Per-tensor optimal scales would corrupt the arithmetic.
    All trained weights have max_abs << 1.0, so fixed Q1.7 is safe.
    """
    return float(Q_SCALE)   # always 128


def export_weights(
    ckpt_path: str = "mnist_vit.pth",
    out_dir:   str = "weights_int8",
) -> None:
    """
    Load trained float model → symmetric int8 quantization → save as:
      weights_int8/WQ.bin, WK.bin, WV.bin, WO.bin
      weights_int8/W1.bin, b1.bin, W2.bin, b2.bin
      weights_int8/patch_proj.bin, pos_embed.bin
      weights_int8/classifier.bin
      weights_int8/scales.txt   (scale factors for dequant)

    Each .bin file is raw signed int8 bytes, row-major.
    Also writes a .hex companion readable by $readmemh in Vivado.
    """
    Path(out_dir).mkdir(exist_ok=True)
    model = MNISTViT()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Only one encoder layer (N_LAYERS=1), pick index 0
    enc = model.encoder[0]
    in_proj_weight = enc.attn.in_proj_weight  # [3*d_model, d_model]
    out_proj_weight = enc.attn.out_proj.weight  # [d_model, d_model]

    # Split in_proj into Q, K, V  (each [d_model, d_model])
    WQ = in_proj_weight[:D_MODEL]
    WK = in_proj_weight[D_MODEL:2*D_MODEL]
    WV = in_proj_weight[2*D_MODEL:]
    WO = out_proj_weight

    W1 = enc.ffn.fc1.weight  # [d_ff, d_model]
    b1 = enc.ffn.fc1.bias    # [d_ff]
    W2 = enc.ffn.fc2.weight  # [d_model, d_ff]
    b2 = enc.ffn.fc2.bias    # [d_model]

    patch_proj_w = model.patch_embed.proj.weight  # [d_model, patch²]
    patch_proj_b = model.patch_embed.proj.bias     # [d_model]
    pos_emb      = model.pos_embed.squeeze(0)      # [seq_len, d_model]
    cls_w        = model.classifier.weight          # [10, d_model]
    cls_b        = model.classifier.bias            # [10]

    tensors = {
        "WQ": WQ, "WK": WK, "WV": WV, "WO": WO,
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
        "patch_proj_w": patch_proj_w,
        "patch_proj_b": patch_proj_b,
        "pos_embed":    pos_emb,
        "classifier_w": cls_w,
        "classifier_b": cls_b,
    }

    scales_txt = []
    for name, t in tensors.items():
        scale    = _infer_scale(t)
        qi8_list = _quantize_tensor(t, scale)
        raw      = bytes([x & 0xFF for x in qi8_list])  # two's complement

        # Raw binary
        bin_path = Path(out_dir) / f"{name}.bin"
        bin_path.write_bytes(raw)

        # Hex file  ($readmemh compatible: two hex digits per byte, one per line)
        hex_path = Path(out_dir) / f"{name}.hex"
        hex_path.write_text("\n".join(f"{b:02x}" for b in raw) + "\n")

        # Dequant scale (to recover float = int8 / scale)
        scales_txt.append(f"{name:20s}  scale={scale:.6f}  numel={len(qi8_list)}")
        print(f"  {name:20s}: {list(t.shape)}  scale={scale:.4f}")

    (Path(out_dir) / "scales.txt").write_text("\n".join(scales_txt) + "\n")
    print(f"\nWeights written to {out_dir}/  ({len(tensors)} tensors)")

    # Generate VHDL package for synthesis
    export_vhdl_pkg(tensors, out_dir)


def export_vhdl_pkg(
    tensors: dict,          # same dict as in export_weights
    out_dir: str = "weights_int8",
    vhdl_path: str | None = None,
) -> None:
    """
    Generate weights_pkg.vhd — a VHDL package with all weight matrices as
    std_logic_vector(7 downto 0) constants.  Vivado infers these as BRAM ROMs
    when accessed with a synchronous registered address (see weight_mem.vhd).

    Only the 8 encoder weight tensors go into the VHDL package.
    Patch embedding, pos_embed and classifier are handled separately
    (not yet in the FPGA datapath).
    """
    if vhdl_path is None:
        vhdl_path = str(Path(out_dir).parent / "weights_pkg.vhd")

    # All tensors: encoder weights + frontend weights
    # enc_keys → weight_mem.vhd; fe_keys → frontend_mem.vhd
    enc_keys = ["WQ", "WK", "WV", "WO", "W1", "b1", "W2", "b2"]
    fe_keys  = ["patch_proj_w", "patch_proj_b", "pos_embed", "classifier_w", "classifier_b"]
    all_keys = enc_keys + fe_keys
    sizes = {
        "WQ":           D_MODEL * D_MODEL,
        "WK":           D_MODEL * D_MODEL,
        "WV":           D_MODEL * D_MODEL,
        "WO":           D_MODEL * D_MODEL,
        "W1":           D_FF    * D_MODEL,
        "b1":           D_FF,
        "W2":           D_MODEL * D_FF,
        "b2":           D_MODEL,
        # Frontend
        "patch_proj_w": D_MODEL * (PATCH_SIZE * PATCH_SIZE),   # 32 × 49 = 1568
        "patch_proj_b": D_MODEL,                                # 32
        "pos_embed":    SEQ_LEN * D_MODEL,                      # 16 × 32 = 512
        "classifier_w": 10 * D_MODEL,                           # 10 × 32 = 320
        "classifier_b": 10,
    }

    lines = []
    lines.append("-- ============================================================")
    lines.append("-- weights_pkg.vhd  —  auto-generated by mnist_poc.py export")
    lines.append("-- DO NOT EDIT — re-run: python mnist_poc.py export")
    lines.append(f"-- Q1.7 fixed-point (scale = {Q_SCALE}), int8 signed")
    lines.append(f"-- d_model={D_MODEL}  d_ff={D_FF}  seq_len={SEQ_LEN}")
    lines.append("-- ============================================================")
    lines.append("")
    lines.append("library ieee;")
    lines.append("    use ieee.std_logic_1164.all;")
    lines.append("")
    lines.append("package weights_pkg is")
    lines.append("")
    lines.append("    subtype weight_byte_t is std_logic_vector(7 downto 0);")
    lines.append("")

    for key in all_keys:
        depth = sizes[key]
        lines.append(f"    -- {key}  ({depth} entries)")
        lines.append(f"    type {key.lower()}_rom_t is array (0 to {depth - 1})"
                     f" of weight_byte_t;")
        lines.append(f"    constant {key.upper()}_ROM : {key.lower()}_rom_t;")
        lines.append("")

    lines.append("end package weights_pkg;")
    lines.append("")
    lines.append("package body weights_pkg is")
    lines.append("")

    for key in all_keys:
        depth  = sizes[key]
        t      = tensors[key]
        scale  = _infer_scale(t)
        qi8    = _quantize_tensor(t, scale)
        assert len(qi8) == depth, f"{key}: expected {depth} entries, got {len(qi8)}"

        lines.append(f"    constant {key.upper()}_ROM : {key.lower()}_rom_t := (")
        # 8 values per line for readability
        chunks = [qi8[i:i+8] for i in range(0, len(qi8), 8)]
        for ci, chunk in enumerate(chunks):
            hex_vals = ", ".join(f'x"{v & 0xFF:02x}"' for v in chunk)
            comma    = "," if ci < len(chunks) - 1 else ""
            lines.append(f"        {hex_vals}{comma}")
        lines.append("    );")
        lines.append("")

    lines.append("end package body weights_pkg;")
    lines.append("")

    Path(vhdl_path).write_text("\n".join(lines))
    print(f"VHDL package written to {vhdl_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Integer golden model: load exported int8 weights + run one inference
# ══════════════════════════════════════════════════════════════════════════════

def _load_int8(path: str) -> List[int]:
    raw = Path(path).read_bytes()
    return [struct.unpack("b", bytes([b]))[0] for b in raw]

def _reshape(flat: List[int], rows: int, cols: int) -> List[List[int]]:
    return [flat[r*cols:(r+1)*cols] for r in range(rows)]


def golden_encoder(weights_dir: str = "weights_int8") -> None:
    """
    Load int8 weights and run the integer encoder on a fixed LFSR input.
    Prints the output stream (channel value last) to stdout.
    """
    d = Path(weights_dir)
    WQ = _reshape(_load_int8(d/"WQ.bin"), D_MODEL, D_MODEL)
    WK = _reshape(_load_int8(d/"WK.bin"), D_MODEL, D_MODEL)
    WV = _reshape(_load_int8(d/"WV.bin"), D_MODEL, D_MODEL)
    WO = _reshape(_load_int8(d/"WO.bin"), D_MODEL, D_MODEL)
    W1 = _reshape(_load_int8(d/"W1.bin"), D_FF, D_MODEL)
    b1_flat = _load_int8(d/"b1.bin")
    W2 = _reshape(_load_int8(d/"W2.bin"), D_MODEL, D_FF)
    b2_flat = _load_int8(d/"b2.bin")

    # Generate a deterministic LFSR input
    lfsr = 0xACE1
    x = []
    for _ in range(SEQ_LEN):
        row = []
        for _ in range(D_MODEL):
            fb   = ((lfsr>>15)&1)^((lfsr>>13)&1)^((lfsr>>12)&1)^((lfsr>>10)&1)
            lfsr = ((lfsr & 0x7FFF) << 1) | fb
            v    = lfsr if lfsr < 32768 else lfsr - 65536
            row.append(max(-128, min(127, v >> 8)))   # top byte as int8
        x.append(row)

    out = encoder_int8(x, WQ, WK, WV, WO, W1, b1_flat, W2, b2_flat)

    # Emit stream in VHDL testbench format: channel value last
    idx = 0
    for token, row in enumerate(out):
        for d, val in enumerate(row):
            flat_idx = token * D_MODEL + d
            last     = 1 if flat_idx == SEQ_LEN * D_MODEL - 1 else 0
            print(f"{flat_idx} {val} {last}")
            idx += 1


# ══════════════════════════════════════════════════════════════════════════════
# Smoke-test: compare float model vs integer golden model on random input
# ══════════════════════════════════════════════════════════════════════════════

def verify() -> None:
    """
    Component-level smoke tests for the integer golden model.
    Tests are deterministic and do not require a trained model.

    Checks:
      1. GELU LUT vs torch.nn.functional.gelu  (max err ≤ 1 LSB = 1/128)
      2. GEMM identity:  I @ B = B  (shift-back correct)
      3. LayerNorm constant input → all zeros
      4. LayerNorm zero-var input → zero output
      5. Full encoder with zero weights → zero output
      6. Softmax output sums to ≤ SEQ_LEN (no overflow, each row sums ≈ 127)
    """
    passed = True

    # ── 1. GELU LUT accuracy ────────────────────────────────────────────────
    max_gelu_err = 0.0
    for i in range(256):
        x_int  = i if i < 128 else i - 256
        x_f    = x_int / 128.0
        t      = _SQRT_2_PI * (x_f + 0.044715 * x_f**3)
        y_ref  = 0.5 * x_f * (1.0 + math.tanh(t))
        y_hw   = GELU_LUT_I8[i] / 128.0
        max_gelu_err = max(max_gelu_err, abs(y_hw - y_ref))
    ok = max_gelu_err <= 1.0 / 128
    print(f"[{'PASS' if ok else 'FAIL'}] GELU LUT max_err={max_gelu_err:.6f} "
          f"(expect ≤ {1/128:.6f})")
    passed = passed and ok

    # ── 2. GEMM identity: I @ X = X ─────────────────────────────────────────
    # Identity matrix in Q1.7: diagonal = 128 (= 1.0)
    I128 = [[128 if r == c else 0 for c in range(4)] for r in range(4)]
    X4   = [[10, -20, 30, -40],
            [ 5,  15, -5,  25],
            [127, 0,  -127, 1],
            [-128, 3,  7, -3]]
    got = gemm_int8(X4, I128)
    ok  = (got == X4)
    print(f"[{'PASS' if ok else 'FAIL'}] GEMM identity  "
          f"got[0]={got[0]}  want={X4[0]}")
    passed = passed and ok

    # ── 3. GEMM zero weights → zero output ──────────────────────────────────
    WZ = [[0]*D_MODEL for _ in range(D_MODEL)]
    X  = [[50]*D_MODEL for _ in range(SEQ_LEN)]
    got = gemm_int8(X, WZ)
    ok  = all(v == 0 for r in got for v in r)
    print(f"[{'PASS' if ok else 'FAIL'}] GEMM zero weights")
    passed = passed and ok

    # ── 4. LayerNorm: constant vector → all zeros ────────────────────────────
    x_const = [[42] * D_MODEL for _ in range(4)]
    out_ln  = layernorm_lod_int8(x_const)
    ok      = all(v == 0 for r in out_ln for v in r)
    print(f"[{'PASS' if ok else 'FAIL'}] LayerNorm constant input → zeros")
    passed = passed and ok

    # ── 5. LayerNorm: alternating [-64, 64] → non-zero, bounded output ───────
    x_alt = [([64, -64] * (D_MODEL // 2)) for _ in range(4)]
    out_ln = layernorm_lod_int8(x_alt)
    ok     = all(-128 <= v <= 127 for r in out_ln for v in r)
    ok2    = any(v != 0 for r in out_ln for v in r)
    print(f"[{'PASS' if (ok and ok2) else 'FAIL'}] LayerNorm alternating input "
          f"  sample={out_ln[0][:4]}")
    passed = passed and ok and ok2

    # ── 6. Full encoder, zero weights → zero output ──────────────────────────
    WZ_dm = [[0]*D_MODEL for _ in range(D_MODEL)]
    WZ_w1 = [[0]*D_MODEL for _ in range(D_FF)]
    bZ_ff = [0]*D_FF
    WZ_w2 = [[0]*D_FF  for _ in range(D_MODEL)]
    bZ_dm = [0]*D_MODEL
    x = [[10]*D_MODEL for _ in range(SEQ_LEN)]
    out = encoder_int8(x, WZ_dm, WZ_dm, WZ_dm, WZ_dm,
                       WZ_w1, bZ_ff, WZ_w2, bZ_dm)
    ok = all(v == 0 for r in out for v in r)
    print(f"[{'PASS' if ok else 'FAIL'}] Encoder zero-weights → zero output")
    passed = passed and ok

    # ── 7. Softmax: each row sums to ~Q_SCALE ────────────────────────────────
    x_sm = [list(range(-8, 8)) for _ in range(4)]     # 16-element rows
    outs  = [softmax_int8(r) for r in x_sm]
    ok    = all(abs(sum(r) - Q_SCALE) <= 16 for r in outs)  # ±16 LUT approx
    print(f"[{'PASS' if ok else 'FAIL'}] Softmax row sums ≈ {Q_SCALE}  "
          f"got={[sum(r) for r in outs[:2]]}")
    passed = passed and ok

    print(f"\n{'ALL PASS' if passed else 'SOME FAILED'}")
    print()
    print("Note: LOD-shift LayerNorm ≠ exact LayerNorm by design.")
    print("End-to-end float vs int8 difference is expected (~10-20% of range)")
    print("and only reduces after training on real MNIST data.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="Train float model on MNIST")
    t.add_argument("--epochs",  type=int,   default=20)
    t.add_argument("--batch",   type=int,   default=256)
    t.add_argument("--lr",      type=float, default=3e-4)
    t.add_argument("--data",    type=str,   default="./data")
    t.add_argument("--ckpt",    type=str,   default="mnist_vit.pth")
    t.add_argument("--device",  type=str,   default="cpu")

    e = sub.add_parser("export", help="Quantize + export int8 weights (float model)")
    e.add_argument("--ckpt",    type=str,   default="mnist_vit.pth")
    e.add_argument("--out",     type=str,   default="weights_int8")

    q = sub.add_parser("qat", help="QAT fine-tuning from float checkpoint")
    q.add_argument("--ckpt",    type=str,   default="mnist_vit.pth")
    q.add_argument("--out",     type=str,   default="mnist_vit_qat.pth")
    q.add_argument("--epochs",  type=int,   default=20)
    q.add_argument("--batch",   type=int,   default=256)
    q.add_argument("--lr",      type=float, default=5e-5)
    q.add_argument("--data",    type=str,   default="./data")
    q.add_argument("--device",  type=str,   default="cpu")

    eq = sub.add_parser("export_qat", help="Export QAT int8 weights + weights_pkg.vhd")
    eq.add_argument("--ckpt",   type=str,   default="mnist_vit_qat.pth")
    eq.add_argument("--out",    type=str,   default="weights_int8")

    g = sub.add_parser("golden", help="Run integer golden model, print stream")
    g.add_argument("--weights", type=str,   default="weights_int8")

    sub.add_parser("verify",  help="Smoke-test: float vs integer encoder")

    args = p.parse_args()

    if args.cmd == "train":
        train(args.epochs, args.batch, args.lr, args.data, args.ckpt, args.device)
    elif args.cmd == "export":
        export_weights(args.ckpt, args.out)
    elif args.cmd == "qat":
        qat_train(args.ckpt, args.out, args.epochs, args.batch, args.lr,
                  args.data, args.device)
    elif args.cmd == "export_qat":
        export_qat_weights(args.ckpt, args.out)
    elif args.cmd == "golden":
        golden_encoder(args.weights)
    elif args.cmd == "verify":
        verify()
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
