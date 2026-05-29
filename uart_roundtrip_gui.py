#!/usr/bin/env python3
"""
uart_roundtrip_gui.py  --  Live MNIST ViT demo dashboard for Basys 3
=====================================================================

Fully automatic: no buttons to press.

Flow per image:
  1. Load MNIST test image
  2. Send 784 raw pixel bytes to FPGA over UART
  3. Receive 0xA5 probe ACK  (FPGA got all 784 bytes, ~0.1 s)
  4. Receive class byte 0-9  (FPGA ran full ViT inference, ~seconds)
  5. Show result, wait HOLD_TIME, then move to next image automatically

Run:
    python uart_roundtrip_gui.py --port COM4
    python uart_roundtrip_gui.py --port COM4 --index 7 --auto-advance
"""

from __future__ import annotations

import argparse
import queue
import struct
import threading
import time
from pathlib import Path
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed.  Run: pip install pyserial")
    sys.exit(1)

# Bit-exact Python reference model (same arithmetic as the VHDL pipeline)
try:
    import golden_model as gm
    _HAVE_GOLDEN = True
except Exception as _exc:          # pragma: no cover
    _HAVE_GOLDEN = False
    print(f"WARNING: golden_model unavailable ({_exc}); Python comparison disabled")

# ── colour palette ────────────────────────────────────────────────────────────
BG      = "#0b1220"
PANEL   = "#111a2e"
CARD    = "#172238"
TEXT    = "#e5eefb"
MUTED   = "#9fb2d1"
ACCENT  = "#4cc9f0"
GOOD    = "#38d39f"
WARN    = "#f5b971"
BAD     = "#ff7a7a"
CORRECT = "#38d39f"   # green  – predicted == true label
WRONG   = "#ff7a7a"   # red    – predicted != true label

PIXEL_SCALE  = 10
IMAGE_SIZE   = 28
IMAGE_BYTES  = IMAGE_SIZE * IMAGE_SIZE
HOLD_TIME_MS = 3000          # ms to show result before advancing

# ── MNIST loader ──────────────────────────────────────────────────────────────

def load_mnist_sample(data_root: Path, index: int) -> tuple[bytes, int]:
    img_path   = data_root / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
    label_path = data_root / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"

    with img_path.open("rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if not (0 <= index < count):
            raise IndexError(f"index {index} out of range (max {count-1})")
        f.seek(16 + index * rows * cols)
        img_bytes = f.read(rows * cols)

    with label_path.open("rb") as f:
        f.read(8)
        f.seek(8 + index)
        label = f.read(1)[0]

    return img_bytes, label


# ── main window ───────────────────────────────────────────────────────────────

class LiveDemo(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.title("Basys 3 — MNIST ViT Live Demo")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.args          = args
        self.q: queue.Queue = queue.Queue()
        self.running       = False
        self.current_index = args.index
        self.sample_pixels: list[int] = [0] * IMAGE_BYTES
        self.sample_label: int = 0
        self.sample_py_pred: int | None = None

        self._build_fonts()
        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(40, self._drain_queue)

        # kick off automatically after window is drawn
        self.after(300, self._run_next)

    # ── fonts ─────────────────────────────────────────────────────────────────

    def _build_fonts(self) -> None:
        self.f_title  = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.f_big    = tkfont.Font(family="Segoe UI", size=64, weight="bold")
        self.f_med    = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.f_label  = tkfont.Font(family="Segoe UI", size=11)
        self.f_log    = tkfont.Font(family="Consolas",  size=9)
        self.f_status = tkfont.Font(family="Segoe UI", size=11, weight="bold")

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = dict(padx=18, pady=10)

        # ── title bar ────────────────────────────────────────────────────────
        title_row = tk.Frame(self, bg=BG)
        title_row.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(title_row, text="Basys 3  ·  MNIST ViT Live Demo",
                 bg=BG, fg=TEXT, font=self.f_title, anchor="w").pack(side="left")
        self.idx_lbl = tk.Label(title_row, text="", bg=BG, fg=MUTED,
                                font=self.f_label, anchor="e")
        self.idx_lbl.pack(side="right")

        # ── main row: image | result ──────────────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="x", padx=18, pady=4)

        # image canvas
        img_card = tk.Frame(main, bg=CARD, highlightthickness=1,
                            highlightbackground="#22314a")
        img_card.pack(side="left", padx=(0, 10))

        tk.Label(img_card, text="Input image", bg=CARD, fg=ACCENT,
                 font=self.f_label).pack(anchor="w", padx=10, pady=(8, 2))

        self.canvas = tk.Canvas(img_card,
                                width=IMAGE_SIZE * PIXEL_SCALE,
                                height=IMAGE_SIZE * PIXEL_SCALE,
                                bg="#050814", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0, 10))

        self.px_items: list[int] = []
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                x0, y0 = c * PIXEL_SCALE, r * PIXEL_SCALE
                self.px_items.append(
                    self.canvas.create_rectangle(
                        x0, y0, x0 + PIXEL_SCALE, y0 + PIXEL_SCALE,
                        outline="", fill="#000000"))

        # true label badge below image
        lbl_row = tk.Frame(img_card, bg=CARD)
        lbl_row.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(lbl_row, text="True label:", bg=CARD, fg=MUTED,
                 font=self.f_label).pack(side="left")
        self.true_lbl_var = tk.StringVar(value="–")
        tk.Label(lbl_row, textvariable=self.true_lbl_var, bg=CARD, fg=TEXT,
                 font=self.f_status).pack(side="left", padx=(6, 0))

        # ── result panel ─────────────────────────────────────────────────────
        res_card = tk.Frame(main, bg=CARD, highlightthickness=1,
                            highlightbackground="#22314a")
        res_card.pack(side="left", fill="both", expand=True)

        # Two big prediction columns: FPGA | Python golden
        preds_row = tk.Frame(res_card, bg=CARD)
        preds_row.pack(pady=(10, 0))

        fpga_col = tk.Frame(preds_row, bg=CARD)
        fpga_col.pack(side="left", padx=18)
        tk.Label(fpga_col, text="FPGA", bg=CARD, fg=ACCENT,
                 font=self.f_label).pack()
        self.pred_var = tk.StringVar(value="–")
        self.pred_lbl = tk.Label(fpga_col, textvariable=self.pred_var,
                                 bg=CARD, fg=TEXT, font=self.f_big, width=2)
        self.pred_lbl.pack()

        py_col = tk.Frame(preds_row, bg=CARD)
        py_col.pack(side="left", padx=18)
        tk.Label(py_col, text="Python", bg=CARD, fg=ACCENT,
                 font=self.f_label).pack()
        self.py_pred_var = tk.StringVar(value="–")
        self.py_pred_lbl = tk.Label(py_col, textvariable=self.py_pred_var,
                                    bg=CARD, fg=MUTED, font=self.f_big, width=2)
        self.py_pred_lbl.pack()

        # FPGA == Python match status (the key "are they the same?" indicator)
        self.match_var = tk.StringVar(value="")
        self.match_lbl = tk.Label(res_card, textvariable=self.match_var,
                                  bg=CARD, fg=MUTED, font=self.f_med)
        self.match_lbl.pack(pady=(2, 0))

        # Correctness vs. ground-truth label
        self.verdict_var = tk.StringVar(value="")
        self.verdict_lbl = tk.Label(res_card, textvariable=self.verdict_var,
                                    bg=CARD, fg=MUTED, font=self.f_label)
        self.verdict_lbl.pack(pady=(0, 8))

        # ── pipeline status ───────────────────────────────────────────────────
        status_card = tk.Frame(self, bg=PANEL, highlightthickness=1,
                               highlightbackground="#22314a")
        status_card.pack(fill="x", padx=18, pady=(6, 0))

        status_inner = tk.Frame(status_card, bg=PANEL)
        status_inner.pack(fill="x", padx=12, pady=8)
        status_inner.columnconfigure(1, weight=1)

        self.stage_var = tk.StringVar(value="Starte…")
        self.stage_lbl = tk.Label(status_inner, textvariable=self.stage_var,
                                  bg=PANEL, fg=ACCENT, font=self.f_status,
                                  width=22, anchor="w")
        self.stage_lbl.grid(row=0, column=0, sticky="w")

        self.pb = ttk.Progressbar(status_inner, orient="horizontal",
                                  mode="determinate", maximum=IMAGE_BYTES,
                                  length=300)
        self.pb.grid(row=0, column=1, sticky="ew", padx=(10, 10))

        self.bytes_var = tk.StringVar(value="")
        tk.Label(status_inner, textvariable=self.bytes_var,
                 bg=PANEL, fg=MUTED, font=self.f_label, width=10,
                 anchor="e").grid(row=0, column=2, sticky="e")

        # ── timing strip ─────────────────────────────────────────────────────
        timing_row = tk.Frame(status_card, bg=PANEL)
        timing_row.pack(fill="x", padx=12, pady=(0, 8))

        self.t_ack_var  = tk.StringVar(value="ACK:  –")
        self.t_inf_var  = tk.StringVar(value="Inference:  –")
        self.t_tot_var  = tk.StringVar(value="Total:  –")
        for var in (self.t_ack_var, self.t_inf_var, self.t_tot_var):
            tk.Label(timing_row, textvariable=var, bg=PANEL, fg=MUTED,
                     font=self.f_label).pack(side="left", padx=(0, 24))

        # ── live log ──────────────────────────────────────────────────────────
        log_card = tk.Frame(self, bg=CARD, highlightthickness=1,
                            highlightbackground="#22314a")
        log_card.pack(fill="both", expand=True, padx=18, pady=(6, 18))

        tk.Label(log_card, text="Live log", bg=CARD, fg=ACCENT,
                 font=self.f_label).pack(anchor="w", padx=10, pady=(6, 2))

        self.log = tk.Text(log_card, height=7, bg="#0b1320", fg=TEXT,
                           font=self.f_log, relief="flat",
                           highlightthickness=0, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log.configure(state="disabled")

        # ttk style
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Horizontal.TProgressbar",
                        troughcolor=PANEL, background=ACCENT)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_stage(self, text: str, colour: str = ACCENT) -> None:
        self.stage_var.set(text)
        self.stage_lbl.configure(fg=colour)

    def _draw_pixels(self, pixels: list[int]) -> None:
        for i, v in enumerate(pixels):
            g = max(0, min(255, v))
            self.canvas.itemconfig(self.px_items[i],
                                   fill=f"#{g:02x}{g:02x}{g:02x}")

    def _emit(self, *item) -> None:
        self.q.put(item)

    # ── queue drain (runs on main thread every 40 ms) ─────────────────────────

    def _drain_queue(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]

                if kind == "stage":
                    self._set_stage(item[1], item[2] if len(item) > 2 else ACCENT)

                elif kind == "progress":
                    sent, total = int(item[1]), int(item[2])
                    self.pb["value"] = sent
                    self.bytes_var.set(f"{sent} / {total}")

                elif kind == "py_pred":
                    py_pred = int(item[1])
                    self.sample_py_pred = py_pred
                    self.py_pred_var.set(str(py_pred))
                    self.py_pred_lbl.configure(fg=TEXT)

                elif kind == "ack":
                    t = float(item[1])
                    self.t_ack_var.set(f"ACK:  {t*1000:.0f} ms")
                    self._log(f"✓ Probe ACK 0xA5 received ({t*1000:.0f} ms)")

                elif kind == "result":
                    pred, t_inf, t_tot = int(item[1]), float(item[2]), float(item[3])
                    true_lbl = self.sample_label
                    py_pred  = self.sample_py_pred
                    ok = pred == true_lbl

                    self.pred_var.set(str(pred))
                    self.pred_lbl.configure(fg=CORRECT if ok else WRONG)

                    # FPGA vs Python golden — the bit-exactness check
                    if py_pred is None:
                        self.match_var.set("Python: n/a")
                        self.match_lbl.configure(fg=MUTED)
                        match_txt = "py=n/a"
                    else:
                        matches = (pred == py_pred)
                        self.py_pred_lbl.configure(fg=GOOD if matches else BAD)
                        self.match_var.set("FPGA = Python ✓" if matches
                                           else "FPGA ≠ Python ✗")
                        self.match_lbl.configure(fg=GOOD if matches else BAD)
                        match_txt = f"py={py_pred} {'MATCH' if matches else 'DIFF'}"

                    self.verdict_var.set(
                        ("✓ correct" if ok else "✗ wrong") + f"  (true {true_lbl})")
                    self.verdict_lbl.configure(fg=CORRECT if ok else WRONG)
                    self.t_inf_var.set(f"Inference:  {t_inf:.2f} s")
                    self.t_tot_var.set(f"Total:  {t_tot:.2f} s")
                    self._log(
                        f"FPGA={pred}  Python={py_pred}  True={true_lbl}  "
                        f"[{match_txt}]  [{'CORRECT' if ok else 'WRONG'}]  "
                        f"inf {t_inf:.2f}s"
                    )
                    self.running = False
                    if self.args.auto_advance:
                        self.after(HOLD_TIME_MS, self._advance)

                elif kind == "error":
                    msg = str(item[1])
                    self._set_stage("Error", BAD)
                    self._log(f"ERROR: {msg}")
                    self.pred_var.set("?")
                    self.pred_lbl.configure(fg=BAD)
                    self.verdict_var.set(msg[:40])
                    self.verdict_lbl.configure(fg=BAD)
                    self.running = False

                elif kind == "log":
                    self._log(str(item[1]))

        except queue.Empty:
            pass

        self.after(40, self._drain_queue)

    # ── control flow ──────────────────────────────────────────────────────────

    def _advance(self) -> None:
        self.current_index = (self.current_index + 1) % 10000
        self._run_next()

    def _run_next(self) -> None:
        if self.running:
            return
        self.running = True

        # reset UI
        self.pred_var.set("–")
        self.pred_lbl.configure(fg=TEXT)
        self.py_pred_var.set("–")
        self.py_pred_lbl.configure(fg=MUTED)
        self.match_var.set("")
        self.sample_py_pred = None
        self.verdict_var.set("")
        self.pb["value"] = 0
        self.bytes_var.set(f"0 / {IMAGE_BYTES}")
        self.t_ack_var.set("ACK:  –")
        self.t_inf_var.set("Inference:  –")
        self.t_tot_var.set("Total:  –")

        # resolve data root
        data_root = Path(self.args.data)
        if not data_root.is_absolute():
            data_root = (Path(__file__).resolve().parent / data_root).resolve()

        # load sample
        try:
            img_bytes, label = load_mnist_sample(data_root, self.current_index)
        except Exception as exc:
            self._emit("error", str(exc))
            return

        self.sample_pixels = list(img_bytes)
        self.sample_label  = label
        self.true_lbl_var.set(str(label))
        self.idx_lbl.configure(
            text=f"Index {self.current_index}  ·  COM{self.args.port.replace('COM','')}")
        self._draw_pixels(self.sample_pixels)
        self._set_stage("Sending…", ACCENT)
        self._log(f"──── Image #{self.current_index}  (true label: {label}) ────")

        # launch worker
        t = threading.Thread(
            target=self._worker,
            args=(self.args.port, bytes(self.sample_pixels)),
            daemon=True)
        t.start()

    def _worker(self, port: str, payload: bytes) -> None:
        try:
            # ── compute Python golden prediction first (cheap, runs off-thread)
            if _HAVE_GOLDEN:
                try:
                    py_pred = gm.predict(list(payload))
                    self._emit("py_pred", py_pred)
                except Exception as exc:
                    self._emit("log", f"Python golden failed: {exc}")

            with serial.Serial(port=port, baudrate=115200,
                               bytesize=8, parity="N", stopbits=1,
                               timeout=self.args.timeout) as ser:
                time.sleep(0.05)
                ser.reset_input_buffer()

                # ── send 784 bytes ────────────────────────────────────────────
                t_send_start = time.time()
                chunk = 64
                sent = 0
                for off in range(0, len(payload), chunk):
                    written = ser.write(payload[off:off + chunk])
                    sent += written
                    self._emit("progress", sent, len(payload))
                ser.flush()

                # ── wait for probe ACK 0xA5 ───────────────────────────────────
                self._emit("stage", "Waiting for ACK…", WARN)
                t_ack_start = time.time()
                ack = ser.read(1)
                t_ack = time.time() - t_ack_start

                if len(ack) != 1:
                    self._emit("error",
                               f"No ACK after {t_ack:.1f}s — is the bitstream loaded?")
                    return
                if ack[0] != 0xA5:
                    self._emit("log",
                               f"Unexpected ACK byte 0x{ack[0]:02X} (expected 0xA5)")

                self._emit("ack", t_ack)

                # ── wait for class result byte ────────────────────────────────
                self._emit("stage", "Inference running…", WARN)
                t_inf_start = time.time()
                res = ser.read(1)
                t_inf = time.time() - t_inf_start
                t_tot = time.time() - t_send_start

                if len(res) != 1:
                    self._emit("error",
                               f"No result byte after {t_inf:.1f}s")
                    return

                pred = res[0]
                if pred > 9:
                    self._emit("log",
                               f"Result byte 0x{pred:02X} out of range — clamping to {pred % 10}")
                    pred = pred % 10

                self._emit("result", pred, t_inf, t_tot)
                self._emit("stage", "Done", GOOD)

        except Exception as exc:
            self._emit("error", str(exc))


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Basys 3 MNIST ViT live demo — fully automatic")
    ap.add_argument("--port",         default="COM4",
                    help="Serial port (e.g. COM4)")
    ap.add_argument("--index",        type=int, default=0,
                    help="MNIST test-set start index (0-9999)")
    ap.add_argument("--data",         default="./data",
                    help="Path to folder containing MNIST/raw/")
    ap.add_argument("--timeout",      type=float, default=30.0,
                    help="Serial read timeout (s) — inference can take several seconds")
    ap.add_argument("--auto-advance", action="store_true",
                    help="Automatically move to next image after result")
    args = ap.parse_args()

    app = LiveDemo(args)
    app.mainloop()


if __name__ == "__main__":
    main()
