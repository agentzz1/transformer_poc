# FPGA MNIST Vision Transformer (ViT) — Basys 3 / Artix-7

**Bit-exact HW/SW co-design · 100% golden-model match across all 10,000 test images · custom QAT pipeline**

This repository contains a fully synthesizable, register-exact **Vision Transformer (ViT) Accelerator** written in **VHDL-2008**, deployed and physically verified on the **Digilent Basys 3 FPGA (Xilinx Artix-7)**. 

By coupling a custom **Quantization-Aware Training (QAT)** pipeline in PyTorch with a register-faithful **Integer Golden Model** and hardware description code, the result is a unified training-to-silicon flow: the physical board produces **bit-exact predictions on all 10,000 MNIST test images** (100% match vs. the integer reference).

---

## Performance & Accuracy

The entire MNIST test set of 10,000 images was evaluated on the physical Basys 3 hardware via UART at 115,200 baud, comparing physical FPGA outputs directly against our Python models.

| Evaluation Domain | Dataset Size | Accuracy | Bit-Exact Matches vs. FPGA |
|-------------------|--------------|----------|----------------------------|
| **PyTorch QAT (Float Emulation)** | 10,000 images | **77.27%** | *N/A (Float vs. Integer)* |
| **Integer Golden Model (Python)** | 10,000 images | **77.21%** | **100.0% (10,000 / 10,000)** |
| **Physical FPGA Hardware (Basys 3)** | 10,000 images | **77.21%** | **100.0% (10,000 / 10,000)** |

> [!NOTE]
> The initial baseline scored **68%** before fixing quantization clipping and a LayerNorm mismatch in the QAT pipeline; the current design reaches **77.21%** with full bit-exactness to the integer reference. 

### FPGA Resource Utilization (xc7a35tcpg236-1)

| Resource | Used | Available | Utilization % |
|----------|------|-----------|---------------|
| **Slice LUTs** | 18,523 | 20,800 | 89.05% |
| **Slice Registers** | 27,245 | 41,600 | 65.49% |
| **Block RAM (BRAM18)** | 6 | 100 | 6.00% |
| **DSPs** | 34 | 90 | 37.78% |

---

## Hardware Architecture & VHDL Sources

The hardware is designed for streaming, pipelined register-transfer level (RTL) execution at **50 MHz**, utilizing internal block RAMs (BRAMs) and DSP slices.

```
Input (784 Pixels)
  │
  ▼
[patch_embed.vhd] (16 Patches of 7x7 -> 16 Tokens of Dim 32 + PosEmbed)
  │
  ▼
[encoder_block.vhd]
  ├── [mha_controller.vhd] (Multi-Head Self-Attention)
  │     └── [softmax.vhd] (256-byte ROM exp-LUT Softmax)
  ├── [residual_add.vhd] (Residual Skip Connection 1)
  ├── [layernorm.vhd] (LOD & Bit-Shift Multiplier-Free LayerNorm)
  │
  ├── [ffn.vhd] (Feed-Forward Network)
  │     └── [psum_activation.vhd] (256-byte ROM GELU-LUT)
  ├── [residual_add.vhd] (Residual Skip Connection 2)
  └── [layernorm.vhd] (Second LayerNorm)
  │
  ▼
[classifier.vhd] (Global Average Pooling -> Linear -> argmax)
  │
  ▼
Output Prediction (7-Segment Display / UART)
```

### VHDL Source Files

| File Basename | Hardware Layer | Mathematical & Register Function |
|:---|:---|:---|
| **[basys3_top.vhd](basys3_top.vhd)** | Top-Level Wrapper | Manages physical board clocks (100MHz PLL to 50MHz), active-low resets, UART RX/TX serial interface (115,200 baud), LED progress indicators, and instantiates the main accelerator core. |
| **[encoder_block.vhd](encoder_block.vhd)** | Transformer Block | Structural top-level connecting Multi-Head Attention, Feed-Forward Network, Residual Additions, and Layer Normalization modules. |
| **[weights_pkg.vhd](weights_pkg.vhd)** | Pre-compiled ROM | **Crucial weight package.** Houses all QAT-trained weights, biases, and positional embeddings represented strictly as pre-compiled signed 8-bit Q1.7 integers. |
| **[patch_embed.vhd](patch_embed.vhd)** | Patch Embedder | Receives 784 raw pixels, partitions them into 16 non-overlapping $7 \times 7$ patches, performs patch projection to $D_{model}=32$ tokens, and adds positional embeddings. |
| **[layernorm.vhd](layernorm.vhd)** | LayerNorm | **Multiplier-free and division-free.** Computes mean and variance, uses Leading-One Detection (LOD) to approximate the reciprocal-square-root, and performs bit-shifts matching `LN_HEADROOM = 2` (4x scale divisor) to keep tokens in Q1.7 boundaries. |
| **[softmax.vhd](softmax.vhd)** | Softmax | Performs numerically-stable integer Softmax. Uses a 256-byte ROM Lookup Table (`_EXP_LUT_Q16`) to compute exp over subtraction differences. |
| **[psum_activation.vhd](psum_activation.vhd)** | GELU LUT | Simulates standard GELU activation in a single clock cycle using a pre-calculated 256-byte ROM Lookup Table (`GELU_LUT_I8`). |
| **[mha_controller.vhd](mha_controller.vhd)** | Self-Attention | Sequences key, query, and value matrix multiplications, computes attention score dot-products, applies Softmax, and outputs the final projected sequence. |
| **[ffn.vhd](ffn.vhd)** | Feed-Forward | Implements the FFN: $FC1$ ($32 \rightarrow 64$), GELU activation, and $FC2$ ($64 \rightarrow 32$). |
| **[gemm_mm.vhd](gemm_mm.vhd)** | active GEMM Engine | **The actual compute core.** Implements a sequential, memory-mapped Multiply-Accumulate (MAC) matrix multiplier. It sequentially sequences calculations over $M \times N \times K$ cycles using highly optimized internal DSP blocks, keeping LUT utilization extremely low. |
| **[classifier.vhd](classifier.vhd)** | Output Classifier | Conducts Global Average Pooling (GAP) over 16 tokens, executes the final GEMM with the classifier's weights/biases, and runs a strict-greater `argmax` to output the final class (0-9). |
| **[seg_test.vhd](seg_test.vhd)** | 7-Segment Multiplexer | Dynamically controls the 4-digit display on the Basys 3 board to display the currently predicted digit. |
| **[control_unit.vhd](control_unit.vhd)** | Controller FSM | Central Finite State Machine that orchestrates the execution states, address generation, RAM writes, and pipelines. |
| *[gemm_os.vhd](gemm_os.vhd)* | Concept Only (Unused) | Leftover reference implementation of an Output-Stationary Systolic Array. Not instantiated in the active hardware design. |
| *[gemm_os_adapter.vhd](gemm_os_adapter.vhd)* | Concept Only (Unused) | Leftover reference adapter for the systolic array. Not instantiated in the active hardware design. |

---

## Python Software & Training Stack

We bridge the gap between continuous floating-point training and discrete integer hardware registers using a three-tier software stack:

1. **[mnist_poc.py](mnist_poc.py) (PyTorch QAT environment):**
   - Incorporates custom PyTorch layers (`HWLayerNorm`, `FQLinear`, `HWSoftmax`, `HWGELU`) using Straight-Through Estimators (STE).
   - Simulates physical integer divisions (`rounding_mode='floor'`) and clamping/saturation (`[-128, 127]`).
   - Implements a **Logit Scaling factor of 8.0** to keep QAT weights small and bounded while allowing PyTorch's loss function to see unconstrained boundaries for gradient flow.
2. **[golden_model.py](golden_model.py) (Software Register Simulator):**
   - Pure Python representation of the hardware. 100% free of PyTorch and floating-point math.
   - Simulates physical memory offsets, exact bitwise shifts (`>> 7`), and Lookup Table indices.
3. **[fpga_vs_python.py](fpga_vs_python.py) (Physical UART Test Suite):**
   - Handles USB-to-UART handshakes at 115,200 baud, sending raw pixels and receiving predictions.
   - Evaluates the accuracy and verifies the bit-exactness of the FPGA board in real time.

---

## Build & Run

### 1. Fine-Tune and Train QAT Model
To train the hardware-identical QAT model on your local PC and export the weights:
```bash
python mnist_poc.py train
```
This runs fine-tuning, achieves **77.27% accuracy**, and writes the weight binaries to `./weights_int8/`.

### 2. Export Weights to VHDL Package
Generate the pre-compiled VHDL ROM package `weights_pkg.vhd`:
```bash
python mnist_poc.py export
```
This updates the Xilinx ROM tables directly inside [weights_pkg.vhd](weights_pkg.vhd).

### 3. Synthesize and Implement via Vivado
Open Xilinx Vivado (2025.2 or similar) and run the batch TCL compilation script to generate the physical bitstream:
```powershell
cd vivado_synth_test
C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat -mode batch -source basys3_impl.tcl
```
This performs RTL synthesis, timing optimization, placement, routing, and generates `basys3_top.bit`.

### 4. Program / Flash the FPGA
Plug the Basys 3 board into your PC via USB, power it on, and program it using the JTAG script:
```powershell
.\flash_transformer.bat
```
The board's LEDs will show programming activity, concluding in `startup status: HIGH`.

### 5. Execute Physical Hardware Evaluation
Run the real-time UART test suite to verify 10,000 images on the physical chip:
```bash
python fpga_vs_python.py --port COM4 --count 10000
```
This streams the images over COM4 and verifies the final, physical **77.21% accuracy** with **100.0% perfect bit-exact matches**!

---

## Architecture Diagrams

Two views of the same architecture:
1. **Mermaid flowchart** — interactive diagram of signals, ports, and submodules, rendered inline by GitHub.
2. **SVG schematic** — a detailed vector schematic with clock domains, ROM access, control paths, and SRAM replay buffers.

---

### 1. Technical VHDL Flowchart (Mermaid)

This vector flowchart is rendered dynamically by GitHub's markdown parser. It contains zero spelling artifacts, exact signal connections, color-coded legends, and is fully searchable:

```mermaid
graph TD
    %% Define Styles & Classes
    classDef cyan fill:#e0f7fa,stroke:#00acc1,stroke-width:2px,color:#006064;
    classDef orange fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#e65100;
    classDef purple fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;
    classDef green fill:#e8f8f5,stroke:#16a085,stroke-width:2px,color:#117864;
    classDef blue fill:#e8f4f8,stroke:#2980b9,stroke-width:2px,color:#1b4f72;

    %% Global Nodes
    Host[Host PC / UART Interface<br>115,200 baud]:::cyan
    Display[4-Digit 7-Segment Display]:::green
    Ack[prediction / done_ack]:::green

    subgraph TopLevel [FPGA Wrapper: basys3_top.vhd]
        Top[Clock Div & UART Control]:::orange
        FrontMem[frontend_mem.vhd<br>UART RX Buffer]:::cyan
        WeightsROM[weights_pkg.vhd<br>Pre-compiled ROM]:::orange
    end

    subgraph Frontend [Pixel & Patch Processing]
        Embed[patch_embed.vhd<br>Patch Projection 16x7x7 -> 16x32]:::cyan
    end

    subgraph Encoder [encoder_block.vhd - Transformer Encoder]
        %% Control Coordinator (positioned at the top)
        CU[control_unit.vhd<br>Central FSM Coordinator]:::blue

        %% First Phase: MHA and its buffers
        input_buf[input_buffer SRAM<br>Stores Skip 1 Connection]:::blue
        
        subgraph MHA_Block [Multi-Head Self-Attention]
            MHA[mha_controller.vhd]:::purple
            GEMM_MM1[gemm_mm.vhd<br>Sequential MAC GEMM<br>Q / K / V / O Projections]:::orange
            Softmax[softmax.vhd<br>exp-LUT Softmax]:::purple
        end
        
        mha_buf[mha_buffer SRAM<br>Stores Attention Outputs]:::blue
        
        %% Second Phase: Residual Addition & LN 1
        subgraph ResAdd_1 [Post-LN Residual block 1]
            Res1[residual_add.vhd<br>Add Elements]:::purple
            LN1[layernorm.vhd<br>Multiplier-Free LayerNorm 1]:::purple
        end
        
        res1_buf[res1_buffer SRAM<br>Stores Skip 2 & FFN Input]:::blue
        
        %% Third Phase: FFN and its buffers
        subgraph FFN_Block [Feed-Forward Network]
            FFN[ffn.vhd]:::purple
            GEMM_MM2[gemm_mm.vhd<br>Sequential MAC GEMM<br>FC1 / FC2 Projections]:::orange
            GELU[psum_activation.vhd<br>GELU LUT]:::purple
        end
        
        ffn_buf[ffn_buffer SRAM<br>Stores FFN Outputs]:::blue

        %% Fourth Phase: Residual Addition & LN 2
        subgraph ResAdd_2 [Post-LN Residual block 2]
            Res2[residual_add.vhd<br>Add Elements]:::purple
            LN2[layernorm.vhd<br>Multiplier-Free LayerNorm 2]:::purple
        end
    end

    subgraph Backend [Output Classification]
        Classifier[classifier.vhd<br>Global Average Pooling]:::green
        FC[Classifier FC Proj<br>Logits 32 -> 10]:::green
        Argmax[Argmax Comparator<br>Selects Class 0-9]:::green
    end

    %% Flow connections
    Host -->|pixel_in| Top
    Top -->|RAW bytes| FrontMem
    FrontMem --> Embed
    
    WeightsROM -->|Projection weights| Embed
    WeightsROM -.->|ROM weights| MHA_Block
    WeightsROM -.->|ROM weights| FFN_Block
    WeightsROM -.->|Classifier weights| Classifier

    %% Encoder Routing
    Embed -->|Tokens x| MHA
    Embed -->|Tokens x| input_buf
    
    %% MHA Internal
    MHA <--> GEMM_MM1
    MHA -.-> Softmax
    MHA -->|Attn Context| mha_buf
    
    %% ResAdd 1
    mha_buf -->|Replay Main| Res1
    input_buf -->|Replay Skip| Res1
    Res1 --> LN1
    LN1 -->|LN output| res1_buf
    
    %% FFN Internal
    res1_buf -->|Replay Main| FFN
    FFN <--> GEMM_MM2
    FFN -.-> GELU
    FFN -->|FFN features| ffn_buf
    
    %% ResAdd 2
    ffn_buf -->|Replay Main| Res2
    res1_buf -->|Replay Skip| Res2
    Res2 --> LN2
    
    %% Backend Routing
    LN2 -->|16 Tokens| Classifier
    Classifier --> FC
    FC --> Argmax
    
    Argmax -->|prediction| Display
    Argmax -->|prediction / done_ack| Ack
    Ack --> Host

    %% Control Lines (conceptual and clean)
    CU -.->|Triggers & Orchestrates| MHA
    CU -.->|Triggers & Orchestrates| FFN
    CU -.->|Controls Replay| Res1
    CU -.->|Controls Replay| Res2
```

* **Legend:** **Cyan** = Activation/Pixel Data | **Orange** = Weights/ROM Access | **Green** = Prediction/Output | **Purple** = Control/Status / Encoder logic | **Blue** = Control Unit & Local SRAM Buffers.

---

### 2. Vector Schematic (SVG)

A detailed SVG schematic of the dataflow, clock domains, weights-ROM access, control paths, and internal SRAM replay buffers — vector, so it stays sharp at any zoom:

![VHDL Architecture Detail Vector Schematic](vhdl_architecture_detail.svg)

