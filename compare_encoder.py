#!/usr/bin/env python3
import sys

def main():
    try:
        with open("encoder_out_mnist.txt", "r") as f:
            py_vals = [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print("[ERROR] encoder_out_mnist.txt not found!")
        sys.exit(1)
        
    try:
        with open("encoder_out_vhdl.txt", "r") as f:
            vhdl_vals = [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print("[ERROR] encoder_out_vhdl.txt not found!")
        sys.exit(1)
        
    print(f"Loaded {len(py_vals)} Python values and {len(vhdl_vals)} VHDL values.")
    
    limit = min(len(py_vals), len(vhdl_vals))
    
    mismatches = 0
    max_diff = 0
    sum_diff = 0
    
    for idx in range(limit):
        diff = abs(py_vals[idx] - vhdl_vals[idx])
        if diff != 0:
            mismatches += 1
            sum_diff += diff
            if diff > max_diff:
                max_diff = diff
            if mismatches <= 20:
                print(f"Mismatch at index {idx:3d} (Token {idx // 32:2d}, Dim {idx % 32:2d}) | Python: {py_vals[idx]:4d} | VHDL: {vhdl_vals[idx]:4d} | Diff: {diff:3d}")
                
    if mismatches > 20:
        print(f"... and {mismatches - 20} more mismatches.")
        
    if limit > 0:
        print("-" * 50)
        print(f"Total elements compared : {limit}")
        print(f"Total mismatches         : {mismatches} ({mismatches / limit * 100:.1f}%)")
        if mismatches > 0:
            print(f"Mean Absolute Error      : {sum_diff / limit:.3f}")
            print(f"Max Absolute Error       : {max_diff}")
        else:
            print("SUCCESS: 100% exact match! 0.00 mean absolute error.")
    else:
        print("[ERROR] No elements to compare.")

if __name__ == "__main__":
    main()
