#!/usr/bin/env python3
"""Integer golden model for the small VHDL transformer testbench.

The model is intentionally fixed-point, not a float transformer with a final
quantize step.  It defines the hardware contract used by the structural VHDL:

  * stream data, weights and biases are signed int16 Q1.15
  * matrix multiplies accumulate products in a wide integer accumulator
  * matmul outputs are arithmetically shifted right by 15 and saturated
  * residual adds saturate to int16
  * FFN activation is currently a pass-through, matching psum_activation.vhd
  * softmax is LUT-based Q1.15
  * LayerNorm uses an integer mean/variance/inv-std approximation

Run GHDL first, then compare one or more captured streams:

  python compare_transformer_reference.py --mode all
  python compare_transformer_reference.py --mode mha
  python compare_transformer_reference.py --mode ffn
  python compare_transformer_reference.py --mode encoder
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


DATA_WIDTH = 16
MODEL_DIM = 32
SEQ_LEN = 16
NUM_HEADS = 8
HEAD_DIM = 4
HIDDEN_DIM = 128
WEIGHT_SCALE = 1024
LFSR_SEED = 0xACE1
Q_SCALE = 1 << (DATA_WIDTH - 1)
TOTAL = SEQ_LEN * MODEL_DIM
LOG_MODEL_DIM = int(math.log2(MODEL_DIM))
LOG_SQRT_HEAD_DIM = int(math.ceil(math.log2(HEAD_DIM))) // 2

I16_MIN = -(1 << 15)
I16_MAX = (1 << 15) - 1

LUT_DEPTH = 256
LUT_X_MIN = -10.0
LUT_MAX_IDX = LUT_DEPTH - 1
EXP_LUT_Q16 = [
    int(math.floor(math.exp(LUT_X_MIN + i * (-LUT_X_MIN / LUT_MAX_IDX)) * (1 << 16) + 0.5))
    for i in range(LUT_DEPTH)
]


@dataclass(frozen=True)
class Target:
    name: str
    vhdl_path: Path
    ref_path: Path
    last_policy: str
    channel_policy: str


TARGETS = {
    "mha": Target("mha", Path("mha_out.txt"), Path("reference_mha_out.txt"), "final", "flat"),
    "ffn": Target("ffn", Path("ffn_out.txt"), Path("reference_ffn_out.txt"), "token", "flat"),
    "encoder": Target("encoder", Path("encoder_out.txt"), Path("reference_encoder_out.txt"), "token", "token"),
}


def to_int16(value: int) -> int:
    value &= 0xFFFF
    if value >= 0x8000:
        value -= 0x10000
    return value


def sat_int16(value: int) -> int:
    if value > I16_MAX:
        return I16_MAX
    if value < I16_MIN:
        return I16_MIN
    return value


def q15_shift(value: int) -> int:
    return value >> (DATA_WIDTH - 1)


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


def weight_matrix(rows: int, cols: int, salt: int) -> list[list[int]]:
    return [[tb_weight(row * cols + col, salt) for col in range(cols)] for row in range(rows)]


def bias_vector(size: int, salt: int) -> list[int]:
    return [tb_bias(i, salt) for i in range(size)]


def transpose(a: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*a)]


def gemm_q15(a: list[list[int]], b: list[list[int]], bias: list[int] | None = None) -> list[list[int]]:
    rows = len(a)
    inner = len(a[0])
    cols = len(b[0])
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc = 0 if bias is None else bias[j] << (DATA_WIDTH - 1)
            for k in range(inner):
                acc += a[i][k] * b[k][j]
            out[i][j] = sat_int16(q15_shift(acc))
    return out


def add_sat(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[sat_int16(x + y) for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def exp_q15_from_diff(diff: int) -> int:
    if diff >= 0:
        return I16_MAX

    magnitude = min(-diff, Q_SCALE)
    scaled = (magnitude * LUT_MAX_IDX) // (10 * Q_SCALE)
    if scaled > LUT_MAX_IDX:
        scaled = LUT_MAX_IDX
    idx = LUT_MAX_IDX - scaled
    return min(I16_MAX, EXP_LUT_Q16[idx] >> 1)


def softmax_q15(row: list[int]) -> list[int]:
    row_max = max(row)
    exps = [exp_q15_from_diff(x - row_max) for x in row]
    denom = sum(exps)
    if denom <= 0:
        return [Q_SCALE // len(row) for _ in row]
    return [sat_int16((x * Q_SCALE) // denom) for x in exps]


def layer_norm_q15(rows: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for row in rows:
        mean = sum(row) >> LOG_MODEL_DIM
        mean_sq = mean * mean
        avg_sq = sum(x * x for x in row) >> LOG_MODEL_DIM
        var = max(1, avg_sq - mean_sq)
        root = math.isqrt(var)
        if root == 0:
            root = 1
        inv_std = min(I16_MAX, (Q_SCALE * Q_SCALE) // root)

        norm_row: list[int] = []
        for x in row:
            norm_row.append(sat_int16(((x - mean) * inv_std) >> (DATA_WIDTH - 1)))
        out.append(norm_row)
    return out


def transformer_int_reference() -> dict[str, list[list[int]]]:
    x = generate_input_int()

    wq = weight_matrix(MODEL_DIM, MODEL_DIM, 1)
    wk = weight_matrix(MODEL_DIM, MODEL_DIM, 2)
    wv = weight_matrix(MODEL_DIM, MODEL_DIM, 3)
    wo = weight_matrix(MODEL_DIM, MODEL_DIM, 4)
    w1 = weight_matrix(HIDDEN_DIM, MODEL_DIM, 5)
    b1 = bias_vector(HIDDEN_DIM, 6)
    w2 = weight_matrix(MODEL_DIM, HIDDEN_DIM, 7)
    b2 = bias_vector(MODEL_DIM, 8)

    concat = [[0 for _ in range(MODEL_DIM)] for _ in range(SEQ_LEN)]
    for head in range(NUM_HEADS):
        start = head * HEAD_DIM
        stop = start + HEAD_DIM
        wq_h = [row[start:stop] for row in wq]
        wk_h = [row[start:stop] for row in wk]
        wv_h = [row[start:stop] for row in wv]

        q = gemm_q15(x, wq_h)
        k = gemm_q15(x, wk_h)
        v = gemm_q15(x, wv_h)

        scores = [[0 for _ in range(SEQ_LEN)] for _ in range(SEQ_LEN)]
        for i in range(SEQ_LEN):
            for j in range(SEQ_LEN):
                acc = 0
                for d in range(HEAD_DIM):
                    acc += q[i][d] * k[j][d]
                scores[i][j] = sat_int16(acc >> ((DATA_WIDTH - 1) + LOG_SQRT_HEAD_DIM))

        probs = [softmax_q15(row) for row in scores]
        ctx = [[0 for _ in range(HEAD_DIM)] for _ in range(SEQ_LEN)]
        for i in range(SEQ_LEN):
            for d in range(HEAD_DIM):
                acc = 0
                for j in range(SEQ_LEN):
                    acc += probs[i][j] * v[j][d]
                ctx[i][d] = sat_int16(q15_shift(acc))

        for token in range(SEQ_LEN):
            concat[token][start:stop] = ctx[token]

    mha = gemm_q15(concat, wo)
    y1 = layer_norm_q15(add_sat(x, mha))

    fc1 = gemm_q15(y1, transpose(w1), b1)
    act = fc1
    ffn = gemm_q15(act, transpose(w2), b2)
    encoder = layer_norm_q15(add_sat(y1, ffn))

    return {"mha": mha, "ffn": ffn, "encoder": encoder}


def flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def expected_channel(idx: int, target: Target) -> int:
    if target.channel_policy == "flat":
        return idx
    if target.channel_policy == "token":
        return idx // MODEL_DIM
    raise ValueError(f"unknown channel policy {target.channel_policy}")


def expected_last(idx: int, target: Target) -> int:
    if target.last_policy == "final":
        return 1 if idx == TOTAL - 1 else 0
    if target.last_policy == "token":
        return 1 if (idx + 1) % MODEL_DIM == 0 else 0
    raise ValueError(f"unknown last policy {target.last_policy}")


def read_stream(path: Path) -> tuple[list[int], list[int], list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run run_ghdl.bat first")

    channels: list[int] = []
    values: list[int] = []
    lasts: list[int] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{lineno}: expected 'channel value last', got {line!r}")
        channel, value, last = map(int, parts)
        channels.append(channel)
        values.append(value)
        lasts.append(last)
    return channels, values, lasts


def protocol_errors(channels: list[int], lasts: list[int], target: Target) -> int:
    errors = 0
    for idx, (channel, last) in enumerate(zip(channels, lasts)):
        if channel != expected_channel(idx, target):
            errors += 1
        if last != expected_last(idx, target):
            errors += 1
    return errors


def write_reference(values: list[int], target: Target) -> None:
    lines = []
    for idx, value in enumerate(values):
        lines.append(f"{expected_channel(idx, target)} {value} {expected_last(idx, target)}")
    target.ref_path.write_text("\n".join(lines) + "\n")


def compare_target(target: Target, ref_matrix: list[list[int]]) -> bool:
    ref_int = flatten(ref_matrix)
    write_reference(ref_int, target)

    channels, vhdl_int, lasts = read_stream(target.vhdl_path)
    if len(vhdl_int) != TOTAL:
        raise ValueError(f"{target.vhdl_path}: expected {TOTAL} outputs, got {len(vhdl_int)}")

    abs_err = [abs(a - b) for a, b in zip(vhdl_int, ref_int)]
    mae = sum(abs_err) / len(abs_err)
    max_abs = max(abs_err)
    exact = sum(1 for a, b in zip(vhdl_int, ref_int) if a == b)
    proto_bad = protocol_errors(channels, lasts, target)
    first_mismatch = next((i for i, (a, b) in enumerate(zip(vhdl_int, ref_int)) if a != b), None)

    print(f"{target.name.upper()} comparison")
    print(f"  vhdl outputs    : {len(vhdl_int)}")
    print(f"  protocol errors : {proto_bad}")
    print(f"  vhdl range      : min={min(vhdl_int)}, max={max(vhdl_int)}, unique={len(set(vhdl_int))}")
    print(f"  ref range       : min={min(ref_int)}, max={max(ref_int)}, unique={len(set(ref_int))}")
    print(f"  exact matches   : {exact}/{len(ref_int)}")
    print(f"  mae             : {mae:.2f} int units ({mae / Q_SCALE:.6f} q15)")
    print(f"  max abs error   : {max_abs} int units ({max_abs / Q_SCALE:.6f} q15)")
    if first_mismatch is not None:
        print(
            "  first mismatch  : "
            f"idx={first_mismatch}, vhdl={vhdl_int[first_mismatch]}, ref={ref_int[first_mismatch]}"
        )
    print(f"  wrote           : {target.ref_path}")

    passed = proto_bad == 0 and max_abs == 0
    print(f"  result          : {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare VHDL structural output against the int golden model.")
    parser.add_argument(
        "--mode",
        choices=["all", "mha", "ffn", "encoder"],
        default="encoder",
        help="Captured stream to compare.",
    )
    args = parser.parse_args()

    refs = transformer_int_reference()
    selected = TARGETS.keys() if args.mode == "all" else [args.mode]

    ok = True
    print("Integer transformer reference")
    print(f"  config          : seq={SEQ_LEN}, model={MODEL_DIM}, heads={NUM_HEADS}, hidden={HIDDEN_DIM}")
    print(f"  q format        : signed int16 Q1.15")
    print()

    for name in selected:
        ok = compare_target(TARGETS[name], refs[name]) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
