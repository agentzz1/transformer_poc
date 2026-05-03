#!/usr/bin/env python3
"""Compare the VHDL encoder simulation against a small float Transformer reference.

The VHDL testbench uses a deliberately small configuration:
  SEQ_LEN=8, MODEL_DIM=32, NUM_HEADS=8, HEAD_DIM=4, HIDDEN_DIM=128.

This script mirrors the testbench stimulus and deterministic test weights, then
computes a conventional post-LayerNorm Transformer encoder block in float.  It is
intended as a numerical sanity check; a mismatch means the VHDL is still a
control/dataflow PoC rather than a numerically faithful Transformer implementation.
"""

from __future__ import annotations

import math
from pathlib import Path


DATA_WIDTH = 16
MODEL_DIM = 32
SEQ_LEN = 8
NUM_HEADS = 8
HEAD_DIM = 4
HIDDEN_DIM = 128
WEIGHT_SCALE = 1024
LFSR_SEED = 0xACE1
Q_SCALE = 2 ** (DATA_WIDTH - 1)

ENCODER_OUT = Path("encoder_out.txt")
REFERENCE_OUT = Path("reference_encoder_out.txt")


def to_int16(value: int) -> int:
    value &= 0xFFFF
    if value >= 0x8000:
        value -= 0x10000
    return value


def clamp_int16(value: int) -> int:
    return max(-32768, min(32767, value))


def tb_weight(addr: int, salt: int) -> int:
    raw = (addr * 37 + salt * 101) % 17
    val = raw - 8
    if val == 0:
        val = 1
    return val * WEIGHT_SCALE


def tb_bias(addr: int, salt: int) -> int:
    raw = (addr * 11 + salt * 23) % 5
    val = raw - 2
    return val * (WEIGHT_SCALE // 4)


def generate_input_int() -> list[list[int]]:
    lfsr = LFSR_SEED
    rows: list[list[int]] = []
    for _token in range(SEQ_LEN):
        row: list[int] = []
        for _elem in range(MODEL_DIM):
            feedback = (
                ((lfsr >> 15) & 1)
                ^ ((lfsr >> 13) & 1)
                ^ ((lfsr >> 12) & 1)
                ^ ((lfsr >> 10) & 1)
            )
            lfsr = ((lfsr & 0x7FFF) << 1) | feedback
            value = to_int16(lfsr)
            if value == 0:
                value = 1
            row.append(value)
        rows.append(row)
    return rows


def as_float_matrix(matrix: list[list[int]]) -> list[list[float]]:
    return [[value / Q_SCALE for value in row] for row in matrix]


def weight_matrix(rows: int, cols: int, salt: int) -> list[list[float]]:
    return [
        [tb_weight(row * cols + col, salt) / Q_SCALE for col in range(cols)]
        for row in range(rows)
    ]


def bias_vector(size: int, salt: int) -> list[float]:
    return [tb_bias(i, salt) / Q_SCALE for i in range(size)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    inner = len(a[0])
    cols = len(b[0])
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            av = a[i][k]
            for j in range(cols):
                out[i][j] += av * b[k][j]
    return out


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def add_bias(a: list[list[float]], b: list[float]) -> list[list[float]]:
    return [[x + b[j] for j, x in enumerate(row)] for row in a]


def softmax_rows(scores: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in scores:
        max_v = max(row)
        exp_row = [math.exp(x - max_v) for x in row]
        denom = sum(exp_row)
        out.append([x / denom for x in exp_row])
    return out


def layer_norm(x: list[list[float]], eps: float = 1.0e-5) -> list[list[float]]:
    out: list[list[float]] = []
    for row in x:
        mean = sum(row) / len(row)
        var = sum((x_i - mean) ** 2 for x_i in row) / len(row)
        inv_std = 1.0 / math.sqrt(var + eps)
        out.append([(x_i - mean) * inv_std for x_i in row])
    return out


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def transformer_reference() -> list[list[float]]:
    x = as_float_matrix(generate_input_int())

    wq = weight_matrix(MODEL_DIM, MODEL_DIM, 1)
    wk = weight_matrix(MODEL_DIM, MODEL_DIM, 2)
    wv = weight_matrix(MODEL_DIM, MODEL_DIM, 3)
    wo = weight_matrix(MODEL_DIM, MODEL_DIM, 4)
    w1 = weight_matrix(HIDDEN_DIM, MODEL_DIM, 5)
    b1 = bias_vector(HIDDEN_DIM, 6)
    w2 = weight_matrix(MODEL_DIM, HIDDEN_DIM, 7)
    b2 = bias_vector(MODEL_DIM, 8)

    concat = [[0.0 for _ in range(MODEL_DIM)] for _ in range(SEQ_LEN)]
    for head in range(NUM_HEADS):
        start = head * HEAD_DIM
        stop = start + HEAD_DIM
        wq_h = [row[start:stop] for row in wq]
        wk_h = [row[start:stop] for row in wk]
        wv_h = [row[start:stop] for row in wv]

        q = matmul(x, wq_h)
        k = matmul(x, wk_h)
        v = matmul(x, wv_h)
        kt = transpose(k)
        scores = matmul(q, kt)
        scale = math.sqrt(HEAD_DIM)
        scores = [[value / scale for value in row] for row in scores]
        probs = softmax_rows(scores)
        ctx = matmul(probs, v)

        for token in range(SEQ_LEN):
            concat[token][start:stop] = ctx[token]

    mha = matmul(concat, wo)
    y1 = layer_norm(add(x, mha))

    fc1 = add_bias(matmul(y1, transpose(w1)), b1)
    act = [[gelu(value) for value in row] for row in fc1]
    ffn = add_bias(matmul(act, transpose(w2)), b2)
    y2 = layer_norm(add(y1, ffn))
    return y2


def quantize_reference(y: list[list[float]]) -> list[int]:
    flat: list[int] = []
    for row in y:
        for value in row:
            flat.append(clamp_int16(round(value * Q_SCALE)))
    return flat


def read_vhdl_output() -> tuple[list[int], list[int], list[int]]:
    if not ENCODER_OUT.exists():
        raise FileNotFoundError(f"{ENCODER_OUT} not found; run run_ghdl.bat first")

    channels: list[int] = []
    values: list[int] = []
    lasts: list[int] = []
    for lineno, line in enumerate(ENCODER_OUT.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{ENCODER_OUT}:{lineno}: expected 'channel value last', got {line!r}")
        channel, value, last = map(int, parts)
        channels.append(channel)
        values.append(value)
        lasts.append(last)
    return channels, values, lasts


def protocol_errors(channels: list[int], lasts: list[int]) -> int:
    errors = 0
    for idx, (channel, last) in enumerate(zip(channels, lasts)):
        expected_channel = idx // MODEL_DIM
        expected_last = 1 if (idx + 1) % MODEL_DIM == 0 else 0
        if channel != expected_channel:
            errors += 1
        if last != expected_last:
            errors += 1
    return errors


def write_reference(ref_int: list[int]) -> None:
    lines = []
    for idx, value in enumerate(ref_int):
        channel = idx // MODEL_DIM
        last = 1 if (idx + 1) % MODEL_DIM == 0 else 0
        lines.append(f"{channel} {value} {last}")
    REFERENCE_OUT.write_text("\n".join(lines) + "\n")


def main() -> int:
    ref_float = transformer_reference()
    ref_int = quantize_reference(ref_float)
    write_reference(ref_int)

    channels, vhdl_int, lasts = read_vhdl_output()
    if len(vhdl_int) != SEQ_LEN * MODEL_DIM:
        raise ValueError(f"expected {SEQ_LEN * MODEL_DIM} VHDL outputs, got {len(vhdl_int)}")

    abs_err = [abs(a - b) for a, b in zip(vhdl_int, ref_int)]
    mae = sum(abs_err) / len(abs_err)
    max_abs = max(abs_err)
    exact = sum(1 for a, b in zip(vhdl_int, ref_int) if a == b)
    proto_bad = protocol_errors(channels, lasts)

    print("Reference comparison")
    print(f"  config          : seq={SEQ_LEN}, model={MODEL_DIM}, heads={NUM_HEADS}, hidden={HIDDEN_DIM}")
    print(f"  vhdl outputs    : {len(vhdl_int)}")
    print(f"  protocol errors : {proto_bad}")
    print(f"  vhdl range      : min={min(vhdl_int)}, max={max(vhdl_int)}, unique={len(set(vhdl_int))}")
    print(f"  ref range       : min={min(ref_int)}, max={max(ref_int)}, unique={len(set(ref_int))}")
    print(f"  exact matches   : {exact}/{len(ref_int)}")
    print(f"  mae             : {mae:.2f} int units ({mae / Q_SCALE:.6f} float)")
    print(f"  max abs error   : {max_abs} int units ({max_abs / Q_SCALE:.6f} float)")
    print(f"  wrote           : {REFERENCE_OUT}")

    if proto_bad == 0 and max_abs == 0:
        print("  result          : PASS")
        return 0

    print("  result          : FAIL - VHDL is not numerically equivalent to the float Transformer reference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
