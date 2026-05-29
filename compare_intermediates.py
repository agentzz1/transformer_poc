#!/usr/bin/env python3
import os

stages = [
    ("Patch Embedding", "py_patch_embed.txt", "pe_out_vhdl.txt"),
    ("MHA Output", "py_mha.txt", "mha_out_vhdl.txt"),
    ("LayerNorm1", "py_y1.txt", "y1_out_vhdl.txt"),
    ("FFN Output", "py_ffn.txt", "ffn_out_vhdl.txt"),
    ("LayerNorm2 (y2)", "encoder_out_mnist.txt", "encoder_out_vhdl.txt")
]

for name, py_file, vhdl_file in stages:
    print("=" * 60)
    print(f"Comparing {name}")
    print("=" * 60)
    
    if not os.path.exists(py_file):
        print(f"[ERROR] Python file {py_file} not found!")
        continue
    if not os.path.exists(vhdl_file):
        print(f"[ERROR] VHDL file {vhdl_file} not found!")
        continue
        
    with open(py_file, "r") as f:
        py_vals = [int(line.strip()) for line in f if line.strip()]
    with open(vhdl_file, "r") as f:
        vhdl_vals = [int(line.strip()) for line in f if line.strip()]
        
    limit = min(len(py_vals), len(vhdl_vals))
    if limit == 0:
        print("[ERROR] No values to compare.")
        continue
        
    print(f"Loaded {len(py_vals)} Python values and {len(vhdl_vals)} VHDL values.")
    
    mismatches = 0
    max_diff = 0
    sum_diff = 0
    first_printed = 0
    
    for idx in range(limit):
        diff = abs(py_vals[idx] - vhdl_vals[idx])
        if diff != 0:
            mismatches += 1
            sum_diff += diff
            if diff > max_diff:
                max_diff = diff
            if first_printed < 10:
                print(f"Mismatch at index {idx:3d} (Token {idx // 32:2d}, Dim {idx % 32:2d}) | Python: {py_vals[idx]:4d} | VHDL: {vhdl_vals[idx]:4d} | Diff: {diff:3d}")
                first_printed += 1
                
    match_pct = (limit - mismatches) / limit * 100
    print("-" * 50)
    print(f"Total elements compared : {limit}")
    print(f"Match Rate              : {match_pct:.2f}% ({limit - mismatches}/{limit})")
    if mismatches > 0:
        print(f"Mean Absolute Error     : {sum_diff / limit:.3f}")
        print(f"Max Absolute Error      : {max_diff}")
    else:
        print("SUCCESS: 100% exact match!")
    print()
