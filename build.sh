#!/bin/bash
# ============================================================================
# BB7 GPU Deciders - Universal Build Script
# Supports: RTX 30-series, 40-series, 50-series, and all CUDA-capable GPUs
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  BB7 GPU Deciders - Build Script"
echo "========================================"
echo ""

# Detect nvcc
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc not found in PATH${NC}"
    echo "Please install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo -n "CUDA version: "
nvcc --version | grep "release" | awk '{print $5, $6}'

# Detect GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}WARNING: No NVIDIA GPUs detected via nvidia-smi${NC}"
else
    echo "Detected GPUs:"
    nvidia-smi -L | sed 's/^/  /'
fi

# Build architecture flags
# Default: native (auto-detect local GPU)
# For distributing binaries to other machines, use multi-arch:
#   ARCH_FLAGS="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_100,code=sm_100"
ARCH_FLAGS="-arch=native"

# Parse arguments
BUILD_QUICK_SIM=true
BUILD_NGRAM_CPS=true
BUILD_TC=true
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH_FLAGS="$2"
            shift 2
            ;;
        --multi-arch)
            # Build for all consumer GPU architectures (30, 40, 50 series)
            ARCH_FLAGS="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_100,code=sm_100"
            echo -e "${BLUE}Multi-arch build for sm_86 (30s), sm_89 (40s), sm_100 (50s)${NC}"
            shift
            ;;
        --quick-sim-only)
            BUILD_NGRAM_CPS=false
            BUILD_TC=false
            shift
            ;;
        --ngram-cps-only)
            BUILD_QUICK_SIM=false
            BUILD_TC=false
            shift
            ;;
        --tc-only)
            BUILD_QUICK_SIM=false
            BUILD_NGRAM_CPS=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --arch <flags>       Override architecture flags (default: -arch=native)"
            echo "  --multi-arch         Build for sm_86, sm_89, sm_100 (for distribution)"
            echo "  --quick-sim-only     Build only gpu_quick_sim"
            echo "  --ngram-cps-only     Build only bb7_ngram_cps"
            echo "  --tc-only            Build only tc_gpu"
            echo "  --verbose            Verbose compilation output"
            echo "  --help               Show this help"
            echo ""
            echo "Architecture flag reference:"
            echo "  -arch=sm_75          Turing (RTX 20-series)"
            echo "  -arch=sm_86          Ampere (RTX 30-series)"
            echo "  -arch=sm_89          Ada Lovelace (RTX 40-series)"
            echo "  -arch=sm_100         Blackwell (RTX 50-series, datacenter)"
            echo "  -arch=sm_120         Blackwell consumer (RTX 50-series)"
            echo "  -arch=native         Auto-detect (default)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

NVCC_FLAGS="-O3 ${ARCH_FLAGS} -std=c++17"

# CUDA 13.x requires standard conforming preprocessor for bb7_ngram_cps
EXTRA_FLAGS=""
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1)
if [ "$CUDA_VERSION" -ge 13 ] 2>/dev/null; then
    EXTRA_FLAGS="-Xcompiler /Zc:preprocessor"
fi
if [ "$VERBOSE" = true ]; then
    NVCC_FLAGS="${NVCC_FLAGS} -v"
fi

echo ""
echo "Compile flags: ${NVCC_FLAGS}"
echo ""

# Build gpu_quick_sim
if [ "$BUILD_QUICK_SIM" = true ]; then
    echo -e "${BLUE}Building gpu_quick_sim...${NC}"
    nvcc ${NVCC_FLAGS} -o gpu_quick_sim gpu_quick_sim.cu
    echo -e "${GREEN}  OK: ./gpu_quick_sim${NC}"
fi

# Build bb7_ngram_cps (needs EXTRA_FLAGS for CUDA 13.x)
if [ "$BUILD_NGRAM_CPS" = true ]; then
    echo -e "${BLUE}Building bb7_ngram_cps...${NC}"
    nvcc ${NVCC_FLAGS} ${EXTRA_FLAGS} -o bb7_ngram_cps bb7_ngram_cps.cu
    echo -e "${GREEN}  OK: ./bb7_ngram_cps${NC}"
fi

# Build tc_gpu
if [ "$BUILD_TC" = true ]; then
    echo -e "${BLUE}Building tc_gpu...${NC}"
    nvcc ${NVCC_FLAGS} -o tc_gpu tc_gpu.cu
    echo -e "${GREEN}  OK: ./tc_gpu${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}  Build complete!${NC}"
echo "========================================"
echo ""
echo "Quick test commands:"
if [ "$BUILD_QUICK_SIM" = true ]; then
    echo "  ./gpu_quick_sim --help"
fi
if [ "$BUILD_NGRAM_CPS" = true ]; then
    echo "  ./bb7_ngram_cps --help"
fi
if [ "$BUILD_TC" = true ]; then
    echo "  ./tc_gpu"
fi
