#!/usr/bin/env python3
"""
send_mnist.py  --  Send an MNIST image to the Basys 3 ViT board over UART
==========================================================================
Usage:
    python send_mnist.py --port COM4 --index 0
    python send_mnist.py --port COM4 --index 5 --show

The script loads MNIST test image #INDEX from the raw IDX files in data/MNIST,
converts it to 784 raw uint8 bytes, and transmits them at 115200 baud to the
Basys 3. In the current board build, the FPGA replies with a probe ACK byte
0xA5 as soon as the 784th byte arrives, then later sends a class byte (0-9)
after inference completes.

After transmission the board will later show the predicted digit on the
7-segment display. The LEDs indicate the pipeline state:
  LED0 - pipeline active
  LED1 - receiving pixels
    LED2 - inference running
    LED3 - result ready (look at the display!)

The ACK byte is a transport check; the later class byte is the classifier result.

Requirements:
    pip install pyserial pillow
"""

import argparse
import struct
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Send MNIST image to Basys 3 ViT")
parser.add_argument("--port",  default="COM4",     help="Serial port (e.g. COM4 or /dev/ttyUSB0)")
parser.add_argument("--baud",  type=int, default=115200, help="Baud rate (must match FPGA)")
parser.add_argument("--index", type=int, default=0,    help="MNIST test-set index 0..9999")
parser.add_argument("--show",  action="store_true",   help="Display the image before sending")
parser.add_argument("--data",  default="./data",       help="Path to MNIST dataset root")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load MNIST test image directly from the raw IDX files
# ---------------------------------------------------------------------------
print(f"Loading MNIST test image #{args.index} ...")

data_root = Path(args.data)
if not data_root.is_absolute():
    data_root = (Path(__file__).resolve().parent / data_root).resolve()
image_path = data_root / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
label_path = data_root / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"

with image_path.open("rb") as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    if args.index < 0 or args.index >= count:
        print(f"ERROR: index {args.index} out of range for {count} test images")
        sys.exit(1)
    image_offset = 16 + args.index * rows * cols
    f.seek(image_offset)
    image_bytes = f.read(rows * cols)

with label_path.open("rb") as f:
    magic_l, count_l = struct.unpack(">II", f.read(8))
    if args.index < 0 or args.index >= count_l:
        print(f"ERROR: index {args.index} out of range for {count_l} labels")
        sys.exit(1)
    label_offset = 8 + args.index
    f.seek(label_offset)
    true_label = f.read(1)[0]

pixels = list(image_bytes)

assert len(pixels) == 784
print(f"  True label : {true_label}")

# ---------------------------------------------------------------------------
# Optionally show the image
# ---------------------------------------------------------------------------
if args.show:
    try:
        from PIL import Image
        img = Image.frombytes("L", (28, 28), image_bytes)
        img = img.resize((280, 280), Image.NEAREST)   # 10x upscale
        img.show()
    except ImportError:
        print("  (pip install pillow to enable image display)")

    # Also print ASCII art
    print("\n  28×28 MNIST image (# = bright, . = dark):")
    for row in range(28):
        line = ""
        for col in range(28):
            p = pixels[row * 28 + col]
            if p > 200:
                line += "##"
            elif p > 100:
                line += "::"
            elif p > 50:
                line += ".."
            else:
                line += "  "
        print("  " + line)
    print()

# ---------------------------------------------------------------------------
# Open serial port
# ---------------------------------------------------------------------------
print(f"Opening {args.port} at {args.baud} baud ...")
try:
    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )
except serial.SerialException as e:
    print(f"ERROR: Cannot open {args.port}: {e}")
    sys.exit(1)

ser.timeout = 5.0

# Brief pause to let the board reset if the port open toggles DTR/RTS
time.sleep(0.1)
ser.reset_input_buffer()

# ---------------------------------------------------------------------------
# Transmit 784 bytes
# ---------------------------------------------------------------------------
payload = bytes(pixels)
print(f"Sending 784 bytes to Basys 3 ...")
t_start = time.time()
ser.write(payload)
ser.flush()
t_sent = time.time()

elapsed_ms = (t_sent - t_start) * 1000
expected_ms = 784 * (1 / args.baud) * 10 * 1000   # 10 bits per byte
print(f"  Sent in {elapsed_ms:.1f} ms  (expected ~{expected_ms:.0f} ms at {args.baud} baud)")

# ---------------------------------------------------------------------------
# Inference timing estimate
# ---------------------------------------------------------------------------
# At 100 MHz:
#   pixel reception:    784 bytes × 8680 cycles = ~6.8 M cycles  (~68 ms)
#   patch_embed:        16 patches × 1664 cycles = ~26.6 K cycles
#   encoder_block:      large but dominated by GEMM (~200 K cycles est.)
#   classifier:         ~1 K cycles
# Total inference time (after last pixel received): << 1 ms at 100 MHz
print()
print("Pixels sent. Watch Basys 3:")
print("  LED1 on   -> receiving pixels")
print("  LED2 on   -> inference running")
print("  LED3 on   -> result ready (class byte on the wire)")
print(f"  Expected answer: {true_label}")
print()

print("Waiting for FPGA ACK (0xA5) ...")
ack = ser.read(1)
if len(ack) != 1:
    print("ERROR: No UART reply received from FPGA")
    ser.close()
    sys.exit(1)

ack_value = ack[0]
if ack_value == 0xA5:
    print("  ACK received: 0xA5 (frame received)")
    ser.timeout = 10.0
    print("Waiting for FPGA result byte (0-9) ...")
    res = ser.read(1)
    if len(res) != 1:
        print("ERROR: No classifier result byte received from FPGA")
        ser.close()
        sys.exit(1)
    result_value = res[0]
else:
    print(f"  WARNING: Expected ACK 0xA5, got 0x{ack_value:02X}")
    result_value = ack_value

print(f"  FPGA class byte: {result_value} (0x{result_value:02X})")
if 0 <= result_value <= 9:
    print(f"  Predicted digit: {result_value}")
else:
    print("  WARNING: Class byte out of range (expected 0-9)")
print()

ser.close()
print("Done.")
