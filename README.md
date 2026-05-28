# BB7 GPU Deciders

CUDA-accelerated deciders for the Busy Beaver BB(7) problem. Three independent tools that run on any NVIDIA GPU with 8GB+ VRAM.

**Supported GPUs**: RTX 30-series, 40-series, 50-series, and all 12.8+ CUDA-capable cards.

---

## Table of Contents

- [What is BB(7)?](#what-is-bb7)
- [The Three Deciders](#the-three-deciders)
- [Hardware Requirements](#hardware-requirements)
- [Safety Guarantees](#safety-guarantees)
- [Quick Start (Windows)](#quick-start-windows)
- [Quick Start (Linux)](#quick-start-linux)
- [Understanding the Output](#understanding-the-output)
- [Getting Real Holdouts Data](#getting-real-holdouts-data)
- [Command Reference](#command-reference)
- [Troubleshooting](#troubleshooting)
- [How It Works](#how-it-works)

---

## What is BB(7)?

The **Busy Beaver** function BB(n) asks: among all n-state, 2-symbol Turing machines, what is the maximum number of steps any halting machine takes?

As of mid-2025, the BB(7) search space has been filtered down to approximately **86 million holdouts** -- candidate machines that no known decider has classified yet. These GPU deciders process holdout files to prove machines non-halting or detect halts.

Learn more: https://bbchallenge.org/

---

## The Three Deciders

| Decider | File | What it does | Speed |
|---------|------|-------------|-------|
| **Translated Cyclers** | `tc_gpu.cu` | Detects translated cycler loops (halt/loop) | ~200-500K TM/s |
| **Quick Sim** | `gpu_quick_sim.cu` | Macro-step simulator to find halts up to 1e14 steps | ~50-200K TM/s |
| **NGramCPS** | `bb7_ngram_cps.cu` | N-gram Closed Position Set proof of non-halt | ~10-50K TM/s |

All three are **independent** -- run any subset in any order. Results accumulate.

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | Any NVIDIA CUDA-capable | RTX 3060+ |
| VRAM | 8GB | 12GB+ for large batches |
| CUDA Toolkit | 11.0+ | 12.8+ for RTX 50-series |

### Per-GPU Memory Usage (estimated)

| Decider | VRAM usage at batch=64 | VRAM usage at batch=256 |
|---------|----------------------|------------------------|
| tc_gpu | ~200MB | ~800MB |
| gpu_quick_sim (8GB mode) | ~400MB | ~1.6GB |
| bb7_ngramcps | auto-scales, ~1-4GB | auto-scales |

---

## Safety Guarantees

**Your hardware will not be damaged.**

- **Out-of-memory**: CUDA returns an error; the program exits. It does not crash your system.
- **Runaway kernels**: Each decider has per-TM device-side timeouts. A single machine cannot hang your GPU indefinitely.
- **GPU watchdog**: Windows and Linux both have GPU watchdog timers (TDR). A kernel running too long triggers a GPU reset -- the screen may flicker, but no hardware damage occurs.
- **Temperature**: All modern GPUs have thermal throttling. If the GPU gets too hot, it slows down automatically. These deciders are compute-heavy but do not stress the GPU more than a modern video game.
- **Batch size auto-calculation**: All three deciders query available free VRAM and compute a safe batch size. They will never allocate more than 85-90% of available memory.
- **OOM retry**: If allocation fails (e.g., another program took VRAM), the decider halves the batch and retries.

---

## Quick Start (Windows)

### Step 1: Install prerequisites

1. **NVIDIA Driver**: https://www.nvidia.com/drivers -- select your GPU, download, install, reboot.

2. **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
   - Operating System: Windows
   - Architecture: x86_64
   - Version: 11/10 (or your Windows version)
   - Installer Type: exe (local)
   - For RTX 50-series (5060/5070/5080/5090), you need **CUDA 12.8 or later**.

3. **Visual Studio Build Tools** (compiler required by CUDA on Windows):
   - Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - Run the installer, check **"Desktop development with C++"**
   - Install

### Step 2: Verify installation

Open **x64 Native Tools Command Prompt for VS 2022** (search in Start menu -- do NOT use regular cmd):

```cmd
nvcc --version
```

You should see `Cuda compilation tools, release 12.x` (or 11.x). Version >= 11.0 works.

```cmd
nvidia-smi
```

You should see your GPU listed with driver version and memory info.

### Step 3: Download this repository

```cmd
cd /d G:\
mkdir bb7_gpu && cd bb7_gpu
git clone https://github.com/YOUR_USERNAME/bb7-gpu-deciders.git .
```

Or download the ZIP and extract to `G:\bb7_gpu`.

### Step 4: Compile

Still in **x64 Native Tools Command Prompt**:

```cmd
cd G:\bb7_gpu

:: tc_gpu and gpu_quick_sim (standard compile)
nvcc -O3 -arch=native -std=c++17 -o tc_gpu.exe tc_gpu.cu
nvcc -O3 -arch=native -std=c++17 -o gpu_quick_sim.exe gpu_quick_sim.cu

:: bb7_ngram_cps (needs extra flag for CUDA 13.x compatibility)
nvcc -O3 -arch=native -std=c++17 -Xcompiler "/Zc:preprocessor" -o bb7_ngram_cps.exe bb7_ngram_cps.cu
```

You will see many `warning C4819` lines from CUDA headers -- these are harmless and can be ignored.

If you are on **CUDA 12.x or older**, the `-Xcompiler "/Zc:preprocessor"` flag is not needed and can be omitted.

If you see `fatal error C1083: Cannot open source file`, make sure you are in the right directory and the `.cu` files exist (`dir` to check).

### Step 5: Create a test input file

Create `test_holdouts.txt` with a few seed strings (one per line):

```
1RB0LD_1LC0RA_1RA1LB_---0RE_1LF0LC_0RG1RB_1LH0RF
1RB0LD_1LC0RA_1RA1LB_---0RE_1LF0LC_0RG1RB_0LH1RF
1RB0LD_1LC0RA_1RA1LB_---0RE_1LF0LC_0RG1RB_1RH0LF
```

### Step 6: Run

```cmd
tc_gpu.exe test_holdouts.txt 0 100000000 2048 30000 my_first_run
```

Arguments explained:
- `test_holdouts.txt` -- input holdouts file
- `0` -- GPU device ID (0 for first/only GPU)
- `100000000` -- step limit (100 million)
- `2048` -- space limit (tape size)
- `30000` -- per-TM timeout in milliseconds (30 seconds)
- `my_first_run` -- output file prefix

Output files created:
- `my_first_run.gpu0.decided.txt` -- machines classified as halt or loop
- `my_first_run.gpu0.unknown.txt` -- machines that exceeded limits

### 8GB GPU users (RTX 3060, 4060, 5060)

For `gpu_quick_sim`, always add `--low-vram`:

```cmd
gpu_quick_sim.exe holdouts.txt 0 my_output 100000000000000 43200 64 --low-vram
```

The `--low-vram` flag halves per-thread memory allocation, keeping total usage well under 8GB.

---

## Quick Start (Linux)

### Step 1: Install CUDA

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

### Step 2: Clone and compile

```bash
git clone https://github.com/YOUR_USERNAME/bb7-gpu-deciders.git
cd bb7-gpu-deciders
chmod +x build.sh
./build.sh
```

Or compile manually:

```bash
nvcc -O3 -arch=native -std=c++17 -o tc_gpu tc_gpu.cu
nvcc -O3 -arch=native -std=c++17 -o gpu_quick_sim gpu_quick_sim.cu
nvcc -O3 -arch=native -std=c++17 -o bb7_ngram_cps bb7_ngram_cps.cu
```

### Step 3: Run

```bash
./tc_gpu test_holdouts.txt 0 100000000 2048 30000 my_first_run
./gpu_quick_sim test_holdouts.txt 0 my_output 100000000000000 43200 64 --low-vram
./bb7_ngram_cps test_holdouts.txt --round1 --round2 --round3
```

---

## Understanding the Output

### Translated Cyclers (`tc_gpu`)

```
[GPU0] 8 TMs | 7.4GB free / 8.5GB total | batch=8 | tl=100000000 | sl=2048 | timeout=30000ms
```

| Field | Meaning |
|-------|---------|
| `8 TMs` | 8 Turing machines to process |
| `7.4GB free` | Available GPU memory |
| `batch=8` | Processing 8 machines at a time (auto-calculated) |
| `tl=100000000` | Step limit: each TM simulated up to 100M steps |
| `sl=2048` | Space limit: tape size of 2048 cells |
| `timeout=30000ms` | Each TM gets 30 seconds before giving up |

Output files:
- `.decided.txt` -- TMs classified (halt or translated cycler loop)
- `.unknown.txt` -- TMs that hit step limit, space limit, or timeout without being classified

### Quick Sim (`gpu_quick_sim`)

Uses a K=32 macro machine: it simulates up to 32 base steps at once when the tape head is at a block edge, falling back to single-stepping inside blocks. This achieves ~10-100x speedup over naive simulation.

Output files:
- `.halt.txt` -- TMs that halted within step limit
- `.unknown.txt` -- TMs that reached max_steps without halting
- `.tape_overflow.txt` -- TMs that exceeded tape allocation

### NGramCPS (`bb7_ngram_cps`)

Runs in 3 rounds with increasing n-gram size (n=2, then n=3, then n=4). Each round filters out machines proven non-halting. Remaining machines pass to the next round.

Output files:
- `.nonhalt.txt` -- TMs proven to never halt (mathematical proof)
- `.unknown_final.txt` -- TMs that survived all rounds

---

## Getting Real Holdouts Data

The `test_holdouts.txt` included here has only a few sample seeds for testing.

For real BB(7) holdouts, visit https://bbchallenge.org/ and download the latest holdouts file (tens of millions of lines, several GB).

Typical workflow with real data:

```bash
# 1. Run NGramCPS first (proves non-halting, fastest filtering)
./bb7_ngram_cps holdouts.txt --round1 --round2 --round3
# Produces: bb7_ngramcps.unknown_final.txt (survivors)

# 2. Run Translated Cyclers on survivors
./tc_gpu bb7_ngramcps.unknown_final.txt 0 100000000 2048 30000 tc_out
# Produces: tc_out.gpu0.unknown.txt (still undecided)

# 3. Run Quick Sim on remaining survivors
./gpu_quick_sim tc_out.gpu0.unknown.txt 0 sim_out 100000000000000 43200 64 --low-vram
# Produces: sim_out.gpu0.halt.txt and sim_out.gpu0.unknown.txt
```

Each step reduces the holdout set. Report any newly classified machines back to bbchallenge.org.

---

## Command Reference

### tc_gpu

```
Usage: ./tc_gpu <holdouts_file> <gpu_id> [tl] [sl] [timeout_ms] [out_prefix]

  holdouts_file  Input file, one TM seed per line
  gpu_id         GPU device index (0 for first GPU)
  tl             Step limit (default: 100M)
  sl             Space/tape limit (default: 2048)
  timeout_ms     Per-TM timeout in milliseconds (default: 0 = none)
  out_prefix     Output file prefix (default: tc_out)
```

### gpu_quick_sim

```
Usage: ./gpu_quick_sim <holdouts_file> <gpu_id> [out_prefix] [max_steps] [timeout_sec] [batch_size] [--low-vram]

  holdouts_file   Input file
  gpu_id          GPU device index
  out_prefix      Output prefix (default: quicksim)
  max_steps       Max steps to simulate (default: 1e14)
  timeout_sec     Wall-clock timeout per batch (default: 43200 = 12h)
  batch_size      0=auto, >0=override
  --low-vram      Halve tape allocation for 8GB GPUs
```

### bb7_ngram_cps

```
Usage: ./bb7_ngram_cps <input_file> [options]

Options:
  --round1, --round2, --round3   Enable/disable rounds (default: all on)
  --gpus N                        Number of GPUs to use (default: all)
  --r1-n N, --r2-n N, --r3-n N   N-gram size per round (default: 2,3,4)
  --r1-contexts N, etc.           Max contexts per round
  --r1-timeout N, etc.            Timeout per round (milliseconds)
  --output-base <path>            Output file prefix
```

---

## Troubleshooting

### "Cannot find compiler 'cl.exe' in PATH"

**Windows**: You opened regular Command Prompt instead of **x64 Native Tools Command Prompt for VS 2022**. Search Start menu for it and use that instead.

### "fatal error C1083: Cannot open source file"

You are not in the directory containing the `.cu` files. Use `cd G:\bb7_gpu` (or wherever you extracted them) and verify with `dir`.

### "Unsupported GPU architecture sm_120" (or sm_89, sm_86)

Your CUDA Toolkit is too old for your GPU.
- RTX 50-series needs **CUDA 12.8+**
- RTX 40-series needs **CUDA 11.8+**
- RTX 30-series needs **CUDA 11.1+**

Download the latest from https://developer.nvidia.com/cuda-downloads

### "CUDA out of memory" or program crashes

- For `gpu_quick_sim`: make sure you added `--low-vram`
- Reduce batch_size manually (e.g., change `64` to `16`)
- Close other GPU-using programs (games, browsers with hardware acceleration, etc.)
- Check with `nvidia-smi` what else is using VRAM

### "class cudaDeviceProp has no member clockRate"

You are using **CUDA 13.x preview**. This field was removed. The code in this repository has been updated to use `cudaDeviceGetAttribute` instead, which works on all CUDA versions. Make sure you have the latest files.

### Long running times

With millions of holdouts, processing can take hours to days. This is normal. The deciders write results incrementally, so you can Ctrl+C to stop and resume later (just filter out already-processed seeds).

### Linux: "nvcc: command not found"

CUDA bin directory is not in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```
Add this line to your `~/.bashrc` to make it permanent.

---

## How It Works

### Architecture: One Process Per GPU

Each decider is designed as a single-process, single-GPU tool. To use multiple GPUs, launch multiple instances with different `gpu_id` values. This avoids CUDA multi-threading context issues that plagued earlier versions.

### Memory Safety

All three deciders follow the same memory safety pattern:

1. Query `cudaMemGetInfo` for free VRAM
2. Compute per-TM memory requirement
3. batch_size = min(free_VRAM * 0.85 / per_TM_memory, BATCH_CAP, total_TMs)
4. If allocation fails, halve batch and retry
5. On 8GB GPUs, `--low-vram` halves tape size for `gpu_quick_sim`

### Timeout Mechanism

All kernels use device-side `clock64()` timing:

```
timeout_cycles = timeout_ms * GPU_clock_rate_kHz
```

Each thread checks elapsed cycles every 1024 iterations. If exceeded, it writes "unknown" and exits. This prevents any single TM from stalling the GPU.

### Algorithm Summaries

**Translated Cyclers**: Records tape snapshots whenever the head visits a new extreme position. Compares current state against past snapshots to detect translated cycles (same pattern shifted by a constant).

**Quick Sim (K=32 Macro Machine)**: Divides the tape into 32-cell blocks. When the head is inside a block, simulates up to 32 steps locally within a 32-bit register. Only accesses global memory when crossing block boundaries.

**NGramCPS**: Builds a closure set of (state, n-gram-context) pairs. If the closure reaches a fixed point without encountering a halting transition, the machine is proven non-halting. Zero false positives.

---

## File Reference

| File | Purpose |
|------|---------|
| `tc_gpu.cu` | Translated Cyclers decider |
| `gpu_quick_sim.cu` | Macro-step halt simulator |
| `bb7_ngram_cps.cu` | NGramCPS non-halt prover |
| `build.sh` | Linux build script (optional, Windows users can ignore) |
| `test_holdouts.txt` | Sample input for testing |

### Do I need build.sh?

- **Linux**: Yes, it makes compilation easier (`./build.sh` compiles all three)
- **Windows**: No, you compile with `nvcc` directly. You can delete `build.sh` if you want.

---
