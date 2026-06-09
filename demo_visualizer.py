#!/usr/bin/env python3
"""
demo_visualizer.py  --  See the MNIST ViT think, then check the board agrees
=============================================================================
Two things in one window:

  1. It runs the bit-exact integer golden model on a MNIST digit and draws every
     stage the hardware goes through: the image split into 16 patches, those
     patches turned into tokens, the attention weights deciding which patches
     matter, the encoder output, and the final ten class scores.  The numbers
     on screen are the SAME integers the VHDL pipeline computes -- this is the
     reference the FPGA was verified against (10,000 / 10,000 exact matches).

  2. Press "Run on FPGA" and the same image is streamed to the Basys 3 over
     UART.  The board's answer appears next to the software answer so you can
     see them line up.

Run it:
    python demo_visualizer.py                 # GUI, FPGA on COM4
    python demo_visualizer.py --port COM5     # GUI, different port
    python demo_visualizer.py --index 42      # start on a specific test image
    python demo_visualizer.py --render fig.png --index 7   # no GUI, save one frame

The board part is optional.  If no board is connected the on-screen result is
still exactly what the chip would output, because the math is identical.
"""
from __future__ import annotations

import argparse
import struct
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import golden_model as gm

# Resolve data relative to the project, not the current working directory.
PROJECT_DIR = Path(gm.__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data" / "MNIST" / "raw"

SEQ, DIM, GRID = gm.SEQ_LEN, gm.D_MODEL, 4   # 16 tokens = 4x4 grid of 7x7 patches


# -- MNIST test set ----------------------------------------------------------
def load_sample(index: int):
    img_path = DATA_DIR / "t10k-images-idx3-ubyte"
    lbl_path = DATA_DIR / "t10k-labels-idx1-ubyte"
    with img_path.open("rb") as f:
        _, count, rows, cols = struct.unpack(">IIII", f.read(16))
        index = index % count
        f.seek(16 + index * rows * cols)
        pixels = list(f.read(rows * cols))
    with lbl_path.open("rb") as f:
        f.seek(8 + index)
        label = f.read(1)[0]
    return pixels, label, index


# -- Instrumented forward pass -----------------------------------------------
# Mirrors golden_model.predict() step for step, but keeps every intermediate.
# It reuses golden_model's own functions, so the arithmetic cannot drift from
# the reference (and from the silicon).
def instrument(pixels):
    W = gm._weights()
    norm = [gm._norm_px(p) for p in pixels]

    x = []
    for p in range(SEQ):
        row = []
        for d in range(DIM):
            acc = W["patch_proj_b"][d] * gm.Q_SCALE
            for k in range(gm.PATCH_SIZE * gm.PATCH_SIZE):
                acc += norm[gm._pix_addr(p, k)] * W["patch_proj_w"][d][k]
            row.append(gm.sat8(gm.sat8(acc >> 7) + W["pos_embed"][p][d]))
        x.append(row)

    Q = gm._gemm_int8(x, gm._transpose(W["WQ"]))
    K = gm._gemm_int8(x, gm._transpose(W["WK"]))
    V = gm._gemm_int8(x, gm._transpose(W["WV"]))

    scores = [[gm.sat8(sum(Q[i][d] * K[j][d] for d in range(DIM)) >> (7 + gm._LOG_SQRT_HD))
               for j in range(SEQ)] for i in range(SEQ)]
    probs = [gm._softmax_int8(r) for r in scores]
    ctx = [[gm.sat8(sum(probs[i][j] * V[j][d] for j in range(SEQ)) >> 7)
            for d in range(DIM)] for i in range(SEQ)]
    mha = gm._gemm_int8(ctx, gm._transpose(W["WO"]))
    y1 = gm._layernorm_lod_int8(gm._add_sat8(x, mha))
    fc1 = gm._gemm_int8(y1, gm._transpose(W["W1"]), W["b1"])
    act = [[gm._gelu_int8(v) for v in row] for row in fc1]
    ffn = gm._gemm_int8(act, gm._transpose(W["W2"]), W["b2"])
    y2 = gm._layernorm_lod_int8(gm._add_sat8(y1, ffn))
    gap = [gm.sat8(sum(y2[t][f] for t in range(SEQ)) >> gm.LOG_SL) for f in range(DIM)]
    logits = gm._gemm_int8([gap], gm._transpose(W["classifier_w"]), W["classifier_b"])[0]

    pred = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[pred]:
            pred = i

    return {
        "norm": np.array(norm).reshape(28, 28),
        "tokens_in": np.array(x),          # 16 x 32
        "attn": np.array(probs),           # 16 x 16
        "tokens_out": np.array(y2),        # 16 x 32
        "logits": np.array(logits),        # 10
        "pred": pred,
    }


def attention_per_patch(attn):
    """Average attention each patch receives, laid back onto the 4x4 patch grid."""
    received = attn.mean(axis=0)                 # mean over query tokens -> 16
    grid = received.reshape(GRID, GRID)
    return np.kron(grid, np.ones((7, 7)))        # upscale 4x4 -> 28x28


# -- Drawing -----------------------------------------------------------------
def draw(fig, res, true_label, fpga_text, fpga_ok):
    fig.clear()
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28,
                          left=0.05, right=0.97, top=0.84, bottom=0.08)

    # 1. input digit with patch grid
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(res["norm"], cmap="gray_r", vmin=-128, vmax=127)
    for g in range(1, GRID):
        ax.axhline(g * 7 - 0.5, color="#4cc9f0", lw=0.8, alpha=0.6)
        ax.axvline(g * 7 - 0.5, color="#4cc9f0", lw=0.8, alpha=0.6)
    ax.set_title(f"Input  ·  16 patches of 7×7  ·  true = {true_label}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 2. where attention goes
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(res["norm"], cmap="gray", vmin=-128, vmax=127)
    ax.imshow(attention_per_patch(res["attn"]), cmap="inferno", alpha=0.55)
    ax.set_title("Attention received per patch", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 3. attention matrix
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(res["attn"], cmap="viridis")
    ax.set_title("Attention weights  (token → token)", fontsize=9)
    ax.set_xlabel("key token"); ax.set_ylabel("query token")
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 4. tokens after patch embedding
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(res["tokens_in"], cmap="coolwarm", vmin=-128, vmax=127, aspect="auto")
    ax.set_title("16 tokens × 32 features  (patch embedding)", fontsize=9)
    ax.set_xlabel("feature"); ax.set_ylabel("token")
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 5. class scores
    ax = fig.add_subplot(gs[1, 1:])
    logits = res["logits"]
    colors = ["#38d39f" if i == res["pred"] else "#9fb2d1" for i in range(10)]
    ax.bar(range(10), logits, color=colors)
    ax.set_xticks(range(10))
    ax.set_title("Class scores (int8 logits)  →  argmax", fontsize=9)
    ax.axhline(0, color="#555", lw=0.6)
    for i, v in enumerate(logits):
        ax.text(i, v + (3 if v >= 0 else -3), str(int(v)),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

    # banner
    pred = res["pred"]
    head = f"Prediction:  {pred}        software (bit-exact reference) = {pred}        {fpga_text}"
    fig.suptitle(head, fontsize=13, fontweight="bold",
                 color=("#1a7f5a" if fpga_ok else "#1d2125"), y=0.95)
    sub = ("The numbers above are the exact integers the FPGA computes — this software model "
           "is what the chip was verified against.")
    fig.text(0.5, 0.89, sub, ha="center", fontsize=8.5, color="#555")


# -- FPGA over UART (same protocol as fpga_vs_python.py) ----------------------
def fpga_predict(port, pixels, timeout=30.0):
    try:
        import serial
    except ImportError:
        return None, "pyserial not installed (pip install pyserial)"
    try:
        with serial.Serial(port, 115200, timeout=timeout) as ser:
            time.sleep(0.1)
            ser.reset_input_buffer()
            ser.write(bytes(p & 0xFF for p in pixels))
            ser.flush()
            ack = ser.read(1)
            if not ack:
                return None, f"no ACK on {port} — is the bitstream flashed?"
            if ack[0] != 0xA5:
                return None, f"unexpected ACK 0x{ack[0]:02X}"
            res = ser.read(1)
            if not res:
                return None, "no result byte (timeout)"
            return res[0] % 10, None
    except Exception as exc:
        return None, str(exc)


# -- GUI ---------------------------------------------------------------------
def run_gui(args):
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("MNIST Vision Transformer — live view + FPGA check")
    root.configure(bg="#f4f6f8")

    state = {"index": args.index, "pixels": None, "label": None,
             "res": None, "fpga_text": "FPGA: press “Run on FPGA”", "fpga_ok": False}
    q: "queue.Queue" = queue.Queue()

    fig = Figure(figsize=(12.5, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=6, pady=6)

    bar = tk.Frame(root, bg="#f4f6f8")
    bar.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

    status = tk.StringVar(value="ready")
    port_var = tk.StringVar(value=args.port)

    def redraw():
        draw(fig, state["res"], state["label"], state["fpga_text"], state["fpga_ok"])
        canvas.draw_idle()

    def load(index):
        pixels, label, index = load_sample(index)
        state.update(index=index, pixels=pixels, label=label,
                     res=instrument(pixels),
                     fpga_text="FPGA: press “Run on FPGA”", fpga_ok=False)
        status.set(f"test image #{index}  (true label {label})")
        redraw()

    def step(delta):
        load(state["index"] + delta)

    def random_img():
        load(int(np.random.randint(0, 10000)))

    def run_fpga():
        status.set(f"sending image #{state['index']} to {port_var.get()} …")
        pixels = list(state["pixels"])
        port = port_var.get()

        def worker():
            t0 = time.time()
            pred, err = fpga_predict(port, pixels)
            q.put((pred, err, time.time() - t0))

        threading.Thread(target=worker, daemon=True).start()

    def poll():
        try:
            pred, err, dt = q.get_nowait()
        except queue.Empty:
            root.after(80, poll)
            return
        sw = state["res"]["pred"]
        if err:
            state["fpga_text"] = f"FPGA: {err}"
            state["fpga_ok"] = False
            status.set(f"FPGA error: {err}")
        else:
            match = (pred == sw)
            state["fpga_text"] = (f"FPGA (board) = {pred}   "
                                  + ("✓ matches software" if match else "✗ differs from software"))
            state["fpga_ok"] = match
            status.set(f"FPGA returned {pred} in {dt:.1f}s  ·  "
                       + ("bit-exact match" if match else "MISMATCH"))
        redraw()
        root.after(80, poll)

    btn = dict(bg="#14506b", fg="white", activebackground="#1d6a8c",
               relief="flat", padx=10, pady=4, font=("Segoe UI", 10))
    tk.Button(bar, text="◀ Prev", command=lambda: step(-1), **btn).pack(side="left", padx=3)
    tk.Button(bar, text="Next ▶", command=lambda: step(1), **btn).pack(side="left", padx=3)
    tk.Button(bar, text="Random", command=random_img, **btn).pack(side="left", padx=3)
    tk.Button(bar, text="Run on FPGA", command=run_fpga,
              bg="#1a7f5a", fg="white", activebackground="#22a06b",
              relief="flat", padx=12, pady=4,
              font=("Segoe UI", 10, "bold")).pack(side="left", padx=(14, 3))
    tk.Label(bar, text="port:", bg="#f4f6f8").pack(side="left", padx=(8, 2))
    tk.Entry(bar, textvariable=port_var, width=7).pack(side="left")
    tk.Label(bar, textvariable=status, bg="#f4f6f8", fg="#333",
             font=("Segoe UI", 10)).pack(side="right", padx=8)

    root.bind("<Right>", lambda e: step(1))
    root.bind("<Left>", lambda e: step(-1))
    root.bind("r", lambda e: random_img())
    root.bind("f", lambda e: run_fpga())

    load(state["index"])
    root.after(80, poll)
    root.mainloop()


# -- headless single-frame render (for testing / slides / backup) ------------
def run_render(args):
    from matplotlib.figure import Figure
    pixels, label, index = load_sample(args.index)
    res = instrument(pixels)
    # sanity: instrumented pass must agree with the canonical golden model
    assert res["pred"] == gm.predict(pixels), "instrumented pass drifted from golden_model!"
    fig = Figure(figsize=(12.5, 7.2), dpi=110)
    fig.patch.set_facecolor("white")
    draw(fig, res, label, "FPGA: (not run in render mode)", False)
    out = Path(args.render)
    fig.savefig(out, bbox_inches="tight")
    print(f"image #{index}: software prediction = {res['pred']}, true = {label}")
    print(f"saved {out.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM4", help="serial port of the Basys 3 (default COM4)")
    ap.add_argument("--index", type=int, default=0, help="MNIST test image index 0-9999")
    ap.add_argument("--render", metavar="PNG", help="render one frame to a file and exit (no GUI)")
    args = ap.parse_args()
    if args.render:
        run_render(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
