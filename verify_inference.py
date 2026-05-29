#!/usr/bin/env python3
import struct
import math
from pathlib import Path
import numpy as np

# Dimensions
PATCH_SIZE  = 7
SEQ_LEN     = 16
D_MODEL     = 32
N_HEADS     = 1
HEAD_DIM    = 32
D_FF        = 64
DATA_WIDTH  = 8
Q_SCALE     = 128
LOG_SL      = 4
_LOG_SQRT_HD = 2  # shift 2 for /4 approximation of sqrt(32)

def sat8(x):
    return max(-128, min(127, x))

def transpose(a):
    return [list(col) for col in zip(*a)]

# Load GELU LUT
_SQRT_2_PI = 0.7978845608028654
def _build_gelu_lut():
    lut = []
    for i in range(256):
        x_int  = i if i < 128 else i - 256
        x_real = x_int / 128.0
        t      = _SQRT_2_PI * (x_real + 0.044715 * x_real**3)
        y_real = 0.5 * x_real * (1.0 + math.tanh(t))
        y_q    = int(y_real * 128.0)
        lut.append(max(-128, min(127, y_q)))
    return lut
GELU_LUT_I8 = _build_gelu_lut()

def gelu_int8(x):
    return GELU_LUT_I8[x & 0xFF]

# Load Softmax LUT
_SM_LUT_DEPTH = 256
_SM_X_MIN     = -10.0
def _build_softmax_lut():
    lut = []
    for i in range(_SM_LUT_DEPTH):
        x_real = _SM_X_MIN + i * (-_SM_X_MIN / (_SM_LUT_DEPTH - 1))
        val    = int(math.floor(math.exp(x_real) * (1 << 16) + 0.5))
        lut.append(val)
    return lut
_EXP_LUT_Q16 = _build_softmax_lut()

def _exp_q7_from_diff(diff):
    if diff >= 0:
        return 127
    magnitude = min(-diff, Q_SCALE)
    scaled    = (magnitude * (_SM_LUT_DEPTH - 1)) // (10 * Q_SCALE)
    idx       = min(_SM_LUT_DEPTH - 1, (_SM_LUT_DEPTH - 1) - scaled)
    return min(127, _EXP_LUT_Q16[idx] >> 9)

def softmax_int8(row):
    row_max = max(row)
    exps    = [_exp_q7_from_diff(x - row_max) for x in row]
    denom   = sum(exps)
    if denom <= 0:
        return [Q_SCALE // len(row)] * len(row)
    return [sat8((e * Q_SCALE) // denom) for e in exps]

# GEMM
def gemm_int8(A, B, bias=None):
    M  = len(A)
    K  = len(A[0])
    N  = len(B[0])
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

# LOD LayerNorm
def _leading_one(x):
    if x == 0:
        return 0
    for i in range(47, -1, -1):
        if (x >> i) & 1:
            return i
    return 0

def layernorm_lod_int8(tokens):
    vb  = 5  # log2(32)
    frac = 7
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
            net = shift - frac
            if net >= 0:
                norm_val = diff >> net
            else:
                norm_val = diff << (-net)
            norm_row.append(sat8(norm_val))
        out.append(norm_row)
    return out

def add_sat8(a, b):
    return [[sat8(x + y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

# Load weight file helpers
def load_weight_bin(path, shape):
    raw = Path(path).read_bytes()
    ints = [struct.unpack("b", bytes([b]))[0] for b in raw]
    assert len(ints) == int(np.prod(shape)), f"Expected {np.prod(shape)} elements in {path}, got {len(ints)}"
    return np.array(ints).reshape(shape).tolist()

# Load all weights
print("Loading weights...")
d = Path("./weights_int8")
WQ = load_weight_bin(d / "WQ.bin", (D_MODEL, D_MODEL))
WK = load_weight_bin(d / "WK.bin", (D_MODEL, D_MODEL))
WV = load_weight_bin(d / "WV.bin", (D_MODEL, D_MODEL))
WO = load_weight_bin(d / "WO.bin", (D_MODEL, D_MODEL))
W1 = load_weight_bin(d / "W1.bin", (D_FF, D_MODEL))
b1 = load_weight_bin(d / "b1.bin", (D_FF,))
W2 = load_weight_bin(d / "W2.bin", (D_MODEL, D_FF))
b2 = load_weight_bin(d / "b2.bin", (D_MODEL,))
patch_proj_w = load_weight_bin(d / "patch_proj_w.bin", (D_MODEL, PATCH_SIZE * PATCH_SIZE))
patch_proj_b = load_weight_bin(d / "patch_proj_b.bin", (D_MODEL,))
pos_embed = load_weight_bin(d / "pos_embed.bin", (SEQ_LEN, D_MODEL))
classifier_w = load_weight_bin(d / "classifier_w.bin", (10, D_MODEL))
classifier_b = load_weight_bin(d / "classifier_b.bin", (10,))

# Load MNIST test images
def load_mnist_samples(data_root, count=100):
    img_path = data_root / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
    label_path = data_root / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"
    
    samples = []
    with img_path.open("rb") as f:
        magic, total_count, rows, cols = struct.unpack(">IIII", f.read(16))
        for idx in range(count):
            img_bytes = f.read(rows * cols)
            samples.append((list(img_bytes), None))
            
    with label_path.open("rb") as f:
        f.read(8)
        for idx in range(count):
            label = f.read(1)[0]
            samples[idx] = (samples[idx][0], label)
            
    return samples

print("Loading MNIST samples...")
samples = load_mnist_samples(Path("./data"), count=20)

# Normalization LUT
def normalize_pixel(p):
    f = (float(p) / 255.0 - 0.1307) / 0.3081 * 128.0
    return sat8(int(round(f)))

# Raster pixel address
def pix_addr(p, k):
    p_row = p // 4
    p_col = p % 4
    r = k // PATCH_SIZE
    c = k % PATCH_SIZE
    return (p_row * PATCH_SIZE + r) * 28 + (p_col * PATCH_SIZE + c)

# Complete model forward pass
def model_forward(pixels):
    # 1. Normalize pixels
    norm_pixels = [normalize_pixel(p) for p in pixels]
    
    # 2. Patch embedding + pos embed
    x = []
    for p in range(SEQ_LEN):
        row = []
        for d_idx in range(D_MODEL):
            acc = patch_proj_b[d_idx] * Q_SCALE
            for k in range(PATCH_SIZE * PATCH_SIZE):
                px_val = norm_pixels[pix_addr(p, k)]
                w_val = patch_proj_w[d_idx][k]
                acc += px_val * w_val
            proj_val = sat8(acc >> 7)
            row.append(sat8(proj_val + pos_embed[p][d_idx]))
        x.append(row)
        
    # 3. Encoder
    # MHA Q/K/V
    Q = gemm_int8(x, transpose(WQ))
    K = gemm_int8(x, transpose(WK))
    V = gemm_int8(x, transpose(WV))
    
    # Scores & softmax
    scores = [[0]*SEQ_LEN for _ in range(SEQ_LEN)]
    for i in range(SEQ_LEN):
        for j in range(SEQ_LEN):
            acc = 0
            for d_idx in range(D_MODEL):
                acc += Q[i][d_idx] * K[j][d_idx]
            scores[i][j] = sat8(acc >> (7 + _LOG_SQRT_HD))
            
    probs = [softmax_int8(row) for row in scores]
    ctx = [[0]*D_MODEL for _ in range(SEQ_LEN)]
    for i in range(SEQ_LEN):
        for d_idx in range(D_MODEL):
            acc = 0
            for j in range(SEQ_LEN):
                acc += probs[i][j] * V[j][d_idx]
            ctx[i][d_idx] = sat8(acc >> 7)
            
    mha = gemm_int8(ctx, transpose(WO))
    y1 = layernorm_lod_int8(add_sat8(x, mha))
    
    # FFN
    fc1 = gemm_int8(y1, transpose(W1), b1)
    act = [[gelu_int8(val) for val in row] for row in fc1]
    ffn = gemm_int8(act, transpose(W2), b2)
    y2 = layernorm_lod_int8(add_sat8(y1, ffn))
    
    # 4. GAP
    gap = []
    for col in transpose(y2):
        gap.append(sat8(sum(col) >> LOG_SL))
        
    # 5. Classifier
    logits = gemm_int8([gap], transpose(classifier_w), classifier_b)[0]
    pred = int(np.argmax(logits))
    return pred, logits

# Run on first 20 samples
correct = 0
for idx, (pixels, true_label) in enumerate(samples):
    pred, logits = model_forward(pixels)
    ok = (pred == true_label)
    if ok:
        correct += 1
    print(f"Image #{idx:2d} | True: {true_label} | Pred: {pred} | {'PASS' if ok else 'FAIL'} | Logits: {logits}")
    
    # Save the very first image's inputs and outputs for GHDL simulation
    if idx == 0:
        # Save normalized pixels
        norm_pixels = [normalize_pixel(p) for p in pixels]
        Path("pixels_norm.txt").write_text("\n".join(str(p) for p in norm_pixels) + "\n")
        
        # Save raw pixels
        Path("pixels_raw.txt").write_text("\n".join(str(p) for p in pixels) + "\n")
        
        # Save encoder outputs (y2 values)
        # We need to run it and intercept y2
        # Let's re-run it
        x = []
        for p in range(SEQ_LEN):
            row = []
            for d_idx in range(D_MODEL):
                acc = patch_proj_b[d_idx] * Q_SCALE
                for k in range(PATCH_SIZE * PATCH_SIZE):
                    px_val = norm_pixels[pix_addr(p, k)]
                    w_val = patch_proj_w[d_idx][k]
                    acc += px_val * w_val
                proj_val = sat8(acc >> 7)
                row.append(sat8(proj_val + pos_embed[p][d_idx]))
            x.append(row)
        Q = gemm_int8(x, transpose(WQ))
        K = gemm_int8(x, transpose(WK))
        V = gemm_int8(x, transpose(WV))
        scores = [[0]*SEQ_LEN for _ in range(SEQ_LEN)]
        for i in range(SEQ_LEN):
            for j in range(SEQ_LEN):
                acc = 0
                for d_idx in range(D_MODEL):
                    acc += Q[i][d_idx] * K[j][d_idx]
                scores[i][j] = sat8(acc >> (7 + _LOG_SQRT_HD))
        probs = [softmax_int8(row) for row in scores]
        ctx = [[0]*D_MODEL for _ in range(SEQ_LEN)]
        for i in range(SEQ_LEN):
            for d_idx in range(D_MODEL):
                acc = 0
                for j in range(SEQ_LEN):
                    acc += probs[i][j] * V[j][d_idx]
                ctx[i][d_idx] = sat8(acc >> 7)
        mha = gemm_int8(ctx, transpose(WO))
        y1 = layernorm_lod_int8(add_sat8(x, mha))
        fc1 = gemm_int8(y1, transpose(W1), b1)
        act = [[gelu_int8(val) for val in row] for row in fc1]
        ffn = gemm_int8(act, transpose(W2), b2)
        y2 = layernorm_lod_int8(add_sat8(y1, ffn))
        
        # Write flat intermediate values
        Path("py_patch_embed.txt").write_text("\n".join(str(v) for row in x for v in row) + "\n")
        Path("py_mha.txt").write_text("\n".join(str(v) for row in mha for v in row) + "\n")
        Path("py_y1.txt").write_text("\n".join(str(v) for row in y1 for v in row) + "\n")
        Path("py_ffn.txt").write_text("\n".join(str(v) for row in ffn for v in row) + "\n")
        Path("encoder_out_mnist.txt").write_text("\n".join(str(v) for row in y2 for v in row) + "\n")
        print(f"\n[INFO] Wrote pixels_norm.txt, pixels_raw.txt, and all intermediate files.")

print(f"\nTotal Correct: {correct}/20 ({correct/20*100:.1f}%)")

