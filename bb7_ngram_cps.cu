/*
 * BB7 NGramCPS Decider - Multi-GPU CUDA
 * NGramCPS (n-gram Closed Position Set) - Zero False Positive
 * Compile: nvcc -O3 -arch=native -std=c++17 -o bb7_ngram_cps bb7_ngram_cps_v3.cu
 * Run:     ./bb7_ngram_cps holdouts.txt --round1 --round2 --round3
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <csignal>

// Tunable GPU Configuration

#ifndef BLOCKS_PER_SM
#define BLOCKS_PER_SM 4
#endif

constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS_PER_SM_CURRENT = BLOCKS_PER_SM;

constexpr int CHECKPOINT_INTERVAL = 10000;
constexpr int PROGRESS_LOG_INTERVAL_SEC = 5;
constexpr double VRAM_SAFETY_FRACTION = 0.80;

// Constants

constexpr int BB7_NUM_STATES = 7;
constexpr int BB7_NUM_SYMBOLS = 2;
constexpr int BB7_MAX_TRANSITIONS = BB7_NUM_STATES * BB7_NUM_SYMBOLS;

constexpr int MAX_N = 5;
constexpr int MAX_NEARBY_BITS = 2 * MAX_N + 1;
constexpr int MAX_LOCAL_CONTEXTS = (1 << 14);
constexpr int MAX_NGRAMS = (1 << MAX_N);

constexpr uint8_t RESULT_NONHALT = 1;
constexpr uint8_t RESULT_HALT    = 2;
constexpr uint8_t RESULT_UNKNOWN = 3;

constexpr uint8_t DIR_LEFT  = 0;
constexpr uint8_t DIR_RIGHT = 1;

constexpr uint8_t STATE_HALT = 255;
constexpr int WORK_QUEUE_DONE = -1;

// Error Checking Macros

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while(0)

#define CUDA_CHECK_LAST(msg) do {                                          \
    cudaError_t err = cudaGetLastError();                                  \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error (%s) at %s:%d: %s\n", msg, __FILE__,  \
                __LINE__, cudaGetErrorString(err));                        \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while(0)

// Data Structures

struct __align__(4) Transition {
    uint8_t write;
    uint8_t move;
    uint8_t next_state;
};

struct __align__(64) TuringMachine {
    Transition trans[BB7_MAX_TRANSITIONS];
};

struct TMResult {
    uint8_t result;
    uint8_t n_used;
    uint32_t closure_size;
    uint32_t ngrams_left_count;
    uint32_t ngrams_right_count;
    uint64_t iterations;
};

// TM Parsing

// Parse a transition string like "1RB", "0LD", "---"
inline bool parse_transition(const char* s, int len, Transition* out) {
    if (len < 3) return false;

    if (s[0] == '-' && s[1] == '-' && s[2] == '-') {
        out->write = 0;
        out->move = DIR_RIGHT;
        out->next_state = STATE_HALT;
        return true;
    }

    char wb = s[0];
    if (wb != '0' && wb != '1') {
        if (wb == '-') {
            out->write = 0;
            out->move = DIR_RIGHT;
            out->next_state = STATE_HALT;
            return true;
        }
        return false;
    }
    out->write = (wb == '1') ? 1 : 0;

    char md = s[1];
    if (md == 'L' || md == 'l') {
        out->move = DIR_LEFT;
    } else if (md == 'R' || md == 'r') {
        out->move = DIR_RIGHT;
    } else {
        return false;
    }

    char ns = s[2];
    if (ns >= 'A' && ns <= 'G') {
        out->next_state = ns - 'A';
    } else if (ns == '-' || ns == 'Z' || ns == 'H') {
        out->next_state = STATE_HALT;
    } else {
        return false;
    }

    return true;
}

// Supports: "1RB0LD_1LC0RA_..." (underscore) or "1RB0LD1LC0RA..." (continuous)
bool parse_tm_seed(const char* seed, TuringMachine* tm, char* normalized_out, int norm_buf_size) {
    memset(tm, 0, sizeof(TuringMachine));
    if (normalized_out && norm_buf_size > 0) {
        normalized_out[0] = '\0';
    }

    int seed_len = (int)strlen(seed);
    char clean[128];
    int clean_len = 0;

    for (int i = 0; i < seed_len && clean_len < 120; i++) {
        char c = seed[i];
        if (c == '_' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            continue;
        }
        clean[clean_len++] = c;
    }
    clean[clean_len] = '\0';

    int transitions_to_parse = BB7_MAX_TRANSITIONS;
    int trans_idx = 0;

    for (int i = 0; i < clean_len - 2 && trans_idx < transitions_to_parse; i += 3) {
        if (!parse_transition(clean + i, 3, &tm->trans[trans_idx])) {
            if (clean[i] == '0' || clean[i] == '1') {
                Transition t;
                t.write = (clean[i] == '1') ? 1 : 0;
                t.move = (clean[i+1] == 'R' || clean[i+1] == 'r') ? DIR_RIGHT : DIR_LEFT;
                if (clean[i+2] >= 'A' && clean[i+2] <= 'G') {
                    t.next_state = clean[i+2] - 'A';
                } else {
                    t.next_state = STATE_HALT;
                }
                tm->trans[trans_idx] = t;
            } else {
                tm->trans[trans_idx].write = 0;
                tm->trans[trans_idx].move = DIR_RIGHT;
                tm->trans[trans_idx].next_state = STATE_HALT;
            }
        }
        trans_idx++;
    }

    for (int i = trans_idx; i < transitions_to_parse; i++) {
        tm->trans[i].write = 0;
        tm->trans[i].move = DIR_RIGHT;
        tm->trans[i].next_state = STATE_HALT;
    }

    if (normalized_out && norm_buf_size > 0) {
        char* p = normalized_out;
        int remaining = norm_buf_size - 1;
        for (int i = 0; i < transitions_to_parse && remaining > 4; i++) {
            if (i > 0 && i % 2 == 0) {
                *p++ = '_';
                remaining--;
            }
            const Transition& t = tm->trans[i];
            if (t.next_state == STATE_HALT) {
                *p++ = '-'; *p++ = '-'; *p++ = '-';
            } else {
                *p++ = t.write ? '1' : '0';
                *p++ = t.move ? 'R' : 'L';
                *p++ = 'A' + t.next_state;
            }
            remaining -= 3;
        }
        *p = '\0';
    }

    return trans_idx > 0;
}

bool tm_has_valid_start(const TuringMachine* tm) {
    return tm->trans[0].next_state != STATE_HALT;
}

int tm_count_defined(const TuringMachine* tm) {
    int count = 0;
    for (int i = 0; i < BB7_MAX_TRANSITIONS; i++) {
        if (tm->trans[i].next_state != STATE_HALT) count++;
    }
    return count;
}

// Device Bitset Utilities

struct BitSet {
    uint32_t* words;
    int num_bits;
    int num_words;
};

__device__ __forceinline__ void bitset_init(BitSet* bs, uint32_t* buffer, int num_bits_val) {
    bs->words = buffer;
    bs->num_bits = num_bits_val;
    bs->num_words = (num_bits_val + 31) / 32;
}

__device__ __forceinline__ void bitset_clear(BitSet* bs) {
    for (int i = 0; i < bs->num_words; i++) {
        bs->words[i] = 0;
    }
}

__device__ __forceinline__ bool bitset_test(const BitSet* bs, int idx) {
    if (idx < 0 || idx >= bs->num_bits) return false;
    int word = idx >> 5;
    int bit = idx & 31;
    return (bs->words[word] >> bit) & 1;
}

__device__ __forceinline__ void bitset_set(BitSet* bs, int idx) {
    if (idx < 0 || idx >= bs->num_bits) return;
    int word = idx >> 5;
    int bit = idx & 31;
    bs->words[word] |= (1u << bit);
}

__device__ __forceinline__ bool bitset_test_and_set(BitSet* bs, int idx) {
    if (idx < 0 || idx >= bs->num_bits) return false;
    int word = idx >> 5;
    int bit = idx & 31;
    uint32_t mask = 1u << bit;
    uint32_t old = bs->words[word];
    if (old & mask) return true;
    bs->words[word] = old | mask;
    return false;
}

// NGramCPS Core Algorithm (Device)

// context encoding: (state << (2*n+1)) | nearby_bits
// nearby_bits layout: bits [2n..n+1] = right n-gram, bit [n] = center, bits [n-1..0] = left n-gram
__device__ __forceinline__ uint32_t get_ngram(uint16_t context, int dir, int n) {
    uint32_t nearby = context & ((1u << (2 * n + 1)) - 1);
    if (dir == DIR_LEFT) {
        return nearby & ((1u << n) - 1);
    } else {
        return (nearby >> (n + 1)) & ((1u << n) - 1);
    }
}

__device__ __forceinline__ uint16_t step_context(
    uint16_t ctx,
    uint8_t write,
    uint8_t next_state,
    uint8_t move_dir,
    uint8_t discovered_bit,
    int n
) {
    uint32_t nearby = ctx & ((1u << (2 * n + 1)) - 1);

    nearby = (nearby & ~(1u << n)) | ((uint32_t)write << n);

    if (move_dir == DIR_LEFT) {
        nearby = ((nearby << 1) | discovered_bit) & ((1u << (2 * n + 1)) - 1);
    } else {
        nearby = (nearby >> 1) | ((uint32_t)discovered_bit << (2 * n));
    }

    return ((uint16_t)next_state << (2 * n + 1)) | nearby;
}

// Core NGramCPS decider. Returns RESULT_NONHALT on closed closure (proof),
// RESULT_UNKNOWN on halt transition, timeout, or resource exhaustion.
__device__ uint8_t ngramcps_decide(
    const TuringMachine* __restrict__ tm,
    int n,
    int max_contexts_allowed,
    int max_iterations,
    unsigned long long timeout_cycles,
    unsigned long long start_clock,
    uint32_t* workspace,
    int workspace_size,
    TMResult* result_out
) {
    const int max_possible_contexts = 1 << (3 + 2 * n + 1);
    const int max_ngrams = 1 << n;
    const int nearby_mask = (1u << (2 * n + 1)) - 1;

    int ctx_words = (max_possible_contexts + 31) / 32;
    int ngram_words = (max_ngrams + 31) / 32;

    if (workspace_size < ctx_words + 2 * ngram_words) {
        if (result_out) {
            result_out->result = RESULT_UNKNOWN;
            result_out->n_used = (uint8_t)n;
            result_out->closure_size = 0;
            result_out->ngrams_left_count = 0;
            result_out->ngrams_right_count = 0;
            result_out->iterations = 0;
        }
        return RESULT_UNKNOWN;
    }

    BitSet contexts, ngrams[2];
    bitset_init(&contexts, workspace, max_possible_contexts);
    bitset_init(&ngrams[DIR_LEFT], workspace + ctx_words, max_ngrams);
    bitset_init(&ngrams[DIR_RIGHT], workspace + ctx_words + ngram_words, max_ngrams);

    bitset_clear(&contexts);
    bitset_clear(&ngrams[DIR_LEFT]);
    bitset_clear(&ngrams[DIR_RIGHT]);

    uint16_t initial_ctx = 0;
    bitset_set(&contexts, initial_ctx);
    bitset_set(&ngrams[DIR_LEFT], 0);
    bitset_set(&ngrams[DIR_RIGHT], 0);

    int num_contexts = 1;
    int num_ngrams[2] = {1, 1};
    unsigned long long iterations = 0;

    while (iterations < (unsigned long long)max_iterations) {
        if (timeout_cycles > 0) {
            unsigned long long elapsed = clock64() - start_clock;
            if (elapsed > timeout_cycles) {
                if (result_out) {
                    result_out->result = RESULT_UNKNOWN;
                    result_out->n_used = (uint8_t)n;
                    result_out->closure_size = (uint32_t)num_contexts;
                    result_out->ngrams_left_count = (uint32_t)num_ngrams[DIR_LEFT];
                    result_out->ngrams_right_count = (uint32_t)num_ngrams[DIR_RIGHT];
                    result_out->iterations = iterations;
                }
                return RESULT_UNKNOWN;
            }
        }

        bool changed = false;
        iterations++;

        for (int ctx_idx = 0; ctx_idx < max_possible_contexts; ctx_idx++) {
            if (!bitset_test(&contexts, ctx_idx)) continue;

            uint16_t ctx = (uint16_t)ctx_idx;
            uint8_t state = ctx >> (2 * n + 1);
            uint32_t nearby = ctx & nearby_mask;
            uint8_t center_bit = (nearby >> n) & 1;

            int trans_idx = state * BB7_NUM_SYMBOLS + center_bit;
            const Transition action = tm->trans[trans_idx];

            // NGramCPS only proves nonhalting, not halting
            if (action.next_state == STATE_HALT) {
                if (result_out) {
                    result_out->result = RESULT_UNKNOWN;
                    result_out->n_used = (uint8_t)n;
                    result_out->closure_size = (uint32_t)num_contexts;
                    result_out->ngrams_left_count = (uint32_t)num_ngrams[DIR_LEFT];
                    result_out->ngrams_right_count = (uint32_t)num_ngrams[DIR_RIGHT];
                    result_out->iterations = iterations;
                }
                return RESULT_UNKNOWN;
            }

            uint8_t move_dir = action.move;
            uint8_t opp_dir = 1 - move_dir;

            uint32_t falling_off = get_ngram(ctx, opp_dir, n);

            if (!bitset_test(&ngrams[opp_dir], falling_off)) {
                bitset_set(&ngrams[opp_dir], falling_off);
                num_ngrams[opp_dir]++;
                changed = true;
            }

            for (uint8_t discovered_bit = 0; discovered_bit <= 1; discovered_bit++) {
                uint16_t new_ctx = step_context(
                    ctx, action.write, action.next_state, move_dir, discovered_bit, n
                );

                uint32_t new_ngram = get_ngram(new_ctx, move_dir, n);

                if (bitset_test(&ngrams[move_dir], new_ngram)) {
                    if (!bitset_test(&contexts, new_ctx)) {
                        if (num_contexts >= max_contexts_allowed) {
                            if (result_out) {
                                result_out->result = RESULT_UNKNOWN;
                                result_out->n_used = (uint8_t)n;
                                result_out->closure_size = (uint32_t)num_contexts;
                                result_out->ngrams_left_count = (uint32_t)num_ngrams[DIR_LEFT];
                                result_out->ngrams_right_count = (uint32_t)num_ngrams[DIR_RIGHT];
                                result_out->iterations = iterations;
                            }
                            return RESULT_UNKNOWN;
                        }
                        bitset_set(&contexts, new_ctx);
                        num_contexts++;
                        changed = true;
                    }
                }
            }
        }

        if (!changed) {
            // Closure reached - mathematical proof of NONHALT
            if (result_out) {
                result_out->result = RESULT_NONHALT;
                result_out->n_used = (uint8_t)n;
                result_out->closure_size = (uint32_t)num_contexts;
                result_out->ngrams_left_count = (uint32_t)num_ngrams[DIR_LEFT];
                result_out->ngrams_right_count = (uint32_t)num_ngrams[DIR_RIGHT];
                result_out->iterations = iterations;
            }
            return RESULT_NONHALT;
        }
    }

    if (result_out) {
        result_out->result = RESULT_UNKNOWN;
        result_out->n_used = (uint8_t)n;
        result_out->closure_size = (uint32_t)num_contexts;
        result_out->ngrams_left_count = (uint32_t)num_ngrams[DIR_LEFT];
        result_out->ngrams_right_count = (uint32_t)num_ngrams[DIR_RIGHT];
        result_out->iterations = iterations;
    }
    return RESULT_UNKNOWN;
}

constexpr int MAX_WORKSPACE_WORDS = (MAX_LOCAL_CONTEXTS + 31) / 32 + 2 * ((MAX_NGRAMS + 31) / 32);

// Work Queue

__device__ int d_work_counter;

// Persistent Kernel (global memory workspace, for n >= 4)
__global__ void ngramcps_persistent_kernel(
    const TuringMachine* __restrict__ tm_list,
    TMResult* __restrict__ results,
    int n,
    int max_contexts,
    int max_iterations,
    unsigned long long timeout_cycles,
    int total_tms,
    uint32_t* workspace_pool,
    int workspace_words_per_thread,
    unsigned long long* processed_counter,
    unsigned long long* nonhalt_counter,
    unsigned long long* unknown_counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    uint32_t* my_workspace = workspace_pool + tid * workspace_words_per_thread;

    unsigned long long local_processed = 0;
    unsigned long long local_nonhalt = 0;
    unsigned long long local_unknown = 0;

    const int BATCH_SIZE = 64;

    while (true) {
        int work_base = atomicAdd(&d_work_counter, BATCH_SIZE);
        if (work_base >= total_tms) break;

        int work_end = (work_base + BATCH_SIZE < total_tms) ? (work_base + BATCH_SIZE) : total_tms;

        for (int idx = work_base + threadIdx.x; idx < work_end; idx += blockDim.x) {
            const TuringMachine* tm = &tm_list[idx];
            TMResult* result = &results[idx];

            unsigned long long start_clock = clock64();

            uint8_t decision = ngramcps_decide(
                tm, n, max_contexts, max_iterations, timeout_cycles,
                start_clock, my_workspace, workspace_words_per_thread, result
            );

            local_processed++;
            if (decision == RESULT_NONHALT) {
                local_nonhalt++;
            } else if (decision == RESULT_UNKNOWN) {
                local_unknown++;
            }
        }

        if ((threadIdx.x & 15) == 0) {
            atomicAdd(processed_counter, local_processed);
            atomicAdd(nonhalt_counter, local_nonhalt);
            atomicAdd(unknown_counter, local_unknown);
            local_processed = 0;
            local_nonhalt = 0;
            local_unknown = 0;
        }
    }

    if (local_processed > 0) {
        atomicAdd(processed_counter, local_processed);
        atomicAdd(nonhalt_counter, local_nonhalt);
        atomicAdd(unknown_counter, local_unknown);
    }
}
__global__ void ngramcps_shared_kernel(
    const TuringMachine* __restrict__ tm_list,
    TMResult* __restrict__ results,
    int n,
    int max_contexts,
    int max_iterations,
    unsigned long long timeout_cycles,
    int total_tms,
    unsigned long long* processed_counter,
    unsigned long long* nonhalt_counter,
    unsigned long long* unknown_counter
) {
    const int max_possible_contexts = 1 << (3 + 2 * n + 1);
    const int max_ngrams = 1 << n;
    const int ctx_words = (max_possible_contexts + 31) / 32;
    const int ngram_words = (max_ngrams + 31) / 32;
    const int total_workspace_words = ctx_words + 2 * ngram_words;

    extern __shared__ uint32_t s_workspace[];

    uint32_t* my_workspace = s_workspace + threadIdx.x * total_workspace_words;

    unsigned long long local_processed = 0;
    unsigned long long local_nonhalt = 0;
    unsigned long long local_unknown = 0;

    const int BATCH_SIZE = 64;

    while (true) {
        int work_base = atomicAdd(&d_work_counter, BATCH_SIZE);
        if (work_base >= total_tms) break;

        int work_end = (work_base + BATCH_SIZE < total_tms) ? (work_base + BATCH_SIZE) : total_tms;

        for (int idx = work_base + threadIdx.x; idx < work_end; idx += blockDim.x) {
            const TuringMachine* tm = &tm_list[idx];
            TMResult* result = &results[idx];

            unsigned long long start_clock = clock64();

            uint8_t decision = ngramcps_decide(
                tm, n, max_contexts, max_iterations, timeout_cycles,
                start_clock, my_workspace, total_workspace_words, result
            );

            local_processed++;
            if (decision == RESULT_NONHALT) local_nonhalt++;
            else if (decision == RESULT_UNKNOWN) local_unknown++;
        }

        if ((threadIdx.x & 15) == 0) {
            atomicAdd(processed_counter, local_processed);
            atomicAdd(nonhalt_counter, local_nonhalt);
            atomicAdd(unknown_counter, local_unknown);
            local_processed = 0;
            local_nonhalt = 0;
            local_unknown = 0;
        }
    }

    if (local_processed > 0) {
        atomicAdd(processed_counter, local_processed);
        atomicAdd(nonhalt_counter, local_nonhalt);
        atomicAdd(unknown_counter, local_unknown);
    }
}

// Host-side Multi-GPU Management

struct GPUResources {
    int gpu_id;
    int num_sms;
    int clock_rate_khz;

    TuringMachine* d_tm_list = nullptr;
    TMResult* d_results = nullptr;
    uint32_t* d_workspace_pool = nullptr;

    unsigned long long* d_processed = nullptr;
    unsigned long long* d_nonhalt = nullptr;
    unsigned long long* d_unknown = nullptr;

    int blocks = 0;
    int threads = 0;
    int total_threads = 0;
    int workspace_words_per_thread = 0;

    cudaStream_t stream = nullptr;
};

__host__ __device__ inline int imin(int a, int b) { return a < b ? a : b; }
__host__ __device__ inline int imax(int a, int b) { return a > b ? a : b; }

// Suggest blocks/SM based on compute capability. Actual value is compile-time default
// unless overridden via -DBLOCKS_PER_SM=N.
int suggest_blocks_per_sm(int major, int minor) {
    int cc = major * 10 + minor;
    if (cc >= 100) return 4;
    if (cc >= 90)  return 4;
    if (cc >= 89)  return 4;
    if (cc >= 80)  return 4;
    if (cc >= 75)  return 4;
    if (cc >= 70)  return 4;
    if (cc >= 60)  return 4;
    if (cc >= 50)  return 4;
    return 2;
}

size_t get_free_vram_bytes() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

void init_gpu(GPUResources* gpu, int gpu_id, int n, int max_contexts) {
    gpu->gpu_id = gpu_id;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    gpu->num_sms = prop.multiProcessorCount;
    cudaDeviceGetAttribute(&gpu->clock_rate_khz, cudaDevAttrClockRate, gpu_id);
    if (gpu->clock_rate_khz == 0) gpu->clock_rate_khz = 2000000;

    printf("[GPU %d] %s, SMs=%d, CC=%d.%d, Clock=%.0f MHz, Mem=%.1f GB\n",
           gpu_id, prop.name, prop.multiProcessorCount,
           prop.major, prop.minor, gpu->clock_rate_khz / 1000.0,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    int suggested_bpsm = suggest_blocks_per_sm(prop.major, prop.minor);
    if (suggested_bpsm != BLOCKS_PER_SM_CURRENT) {
        printf("[GPU %d] Hint: For CC %d.%d, consider -DBLOCKS_PER_SM=%d (currently %d)\n",
               gpu_id, prop.major, prop.minor, suggested_bpsm, BLOCKS_PER_SM_CURRENT);
    }

    gpu->threads = THREADS_PER_BLOCK;
    gpu->blocks = gpu->num_sms * BLOCKS_PER_SM_CURRENT;
    gpu->total_threads = gpu->blocks * gpu->threads;

    int max_possible_contexts = 1 << (3 + 2 * n + 1);
    int max_ngrams = 1 << n;
    int ctx_words = (max_possible_contexts + 31) / 32;
    int ngram_words = (max_ngrams + 31) / 32;
    gpu->workspace_words_per_thread = ctx_words + 2 * ngram_words;

    CUDA_CHECK(cudaStreamCreate(&gpu->stream));

    CUDA_CHECK(cudaMalloc(&gpu->d_processed, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&gpu->d_nonhalt, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&gpu->d_unknown, sizeof(unsigned long long)));

    printf("[GPU %d] Config: blocks=%d, threads=%d, total_threads=%d, ws_words/thread=%d\n",
           gpu_id, gpu->blocks, gpu->threads, gpu->total_threads,
           gpu->workspace_words_per_thread);
}

void cleanup_gpu(GPUResources* gpu) {
    CUDA_CHECK(cudaSetDevice(gpu->gpu_id));
    if (gpu->d_tm_list) cudaFree(gpu->d_tm_list);
    if (gpu->d_results) cudaFree(gpu->d_results);
    if (gpu->d_workspace_pool) cudaFree(gpu->d_workspace_pool);
    if (gpu->d_processed) cudaFree(gpu->d_processed);
    if (gpu->d_nonhalt) cudaFree(gpu->d_nonhalt);
    if (gpu->d_unknown) cudaFree(gpu->d_unknown);
    if (gpu->stream) cudaStreamDestroy(gpu->stream);
}

void launch_kernel_on_gpu(
    GPUResources* gpu,
    const std::vector<TuringMachine>& tm_list,
    int n,
    int max_contexts,
    int max_iterations,
    int timeout_per_tm_ms
) {
    CUDA_CHECK(cudaSetDevice(gpu->gpu_id));

    int total_tms = (int)tm_list.size();

    size_t tm_bytes = total_tms * sizeof(TuringMachine);
    size_t result_bytes = total_tms * sizeof(TMResult);

    CUDA_CHECK(cudaMalloc(&gpu->d_tm_list, tm_bytes));
    CUDA_CHECK(cudaMalloc(&gpu->d_results, result_bytes));

    CUDA_CHECK(cudaMemcpyAsync(gpu->d_tm_list, tm_list.data(), tm_bytes,
                               cudaMemcpyHostToDevice, gpu->stream));

    // clockRate is in kHz, so timeout_cycles = ms * kHz
    unsigned long long timeout_cycles = 0;
    if (timeout_per_tm_ms > 0 && gpu->clock_rate_khz > 0) {
        timeout_cycles = (unsigned long long)timeout_per_tm_ms * (unsigned long long)gpu->clock_rate_khz;
    }

    if (n >= 4) {
        size_t pool_bytes = (size_t)gpu->total_threads *
                            gpu->workspace_words_per_thread * sizeof(uint32_t);

        size_t free_vram = get_free_vram_bytes();
        size_t max_allowed_pool = (size_t)(free_vram * VRAM_SAFETY_FRACTION);

        if (pool_bytes > max_allowed_pool && gpu->blocks > gpu->num_sms) {
            int new_blocks = imax(gpu->num_sms,
                                  (int)((double)gpu->blocks * max_allowed_pool / pool_bytes));
            if (new_blocks < gpu->blocks) {
                printf("[GPU %d] WARNING: Workspace pool (%.1f MB) exceeds %.0f%% of free VRAM (%.1f MB).\n",
                       gpu->gpu_id, pool_bytes / (1024.0 * 1024.0),
                       VRAM_SAFETY_FRACTION * 100.0,
                       free_vram / (1024.0 * 1024.0));
                printf("[GPU %d] Auto-scaling: blocks %d -> %d (%.1f MB)\n",
                       gpu->gpu_id, gpu->blocks, new_blocks,
                       (double)new_blocks * gpu->threads * gpu->workspace_words_per_thread
                       * sizeof(uint32_t) / (1024.0 * 1024.0));
                gpu->blocks = new_blocks;
                gpu->total_threads = gpu->blocks * gpu->threads;
                pool_bytes = (size_t)gpu->total_threads *
                             gpu->workspace_words_per_thread * sizeof(uint32_t);
            }
        }

        printf("[GPU %d] Allocating workspace pool: %.1f MB (timeout=%llucycles at %d kHz)\n",
               gpu->gpu_id, pool_bytes / (1024.0 * 1024.0),
               timeout_cycles, gpu->clock_rate_khz);
        CUDA_CHECK(cudaMalloc(&gpu->d_workspace_pool, pool_bytes));
        CUDA_CHECK(cudaMemsetAsync(gpu->d_workspace_pool, 0, pool_bytes, gpu->stream));
    }

    CUDA_CHECK(cudaMemsetAsync(gpu->d_processed, 0, sizeof(unsigned long long), gpu->stream));
    CUDA_CHECK(cudaMemsetAsync(gpu->d_nonhalt, 0, sizeof(unsigned long long), gpu->stream));
    CUDA_CHECK(cudaMemsetAsync(gpu->d_unknown, 0, sizeof(unsigned long long), gpu->stream));

    int h_zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_work_counter, &h_zero, sizeof(int),
                                        0, cudaMemcpyHostToDevice, gpu->stream));

    if (n <= 3) {
        int max_possible_contexts = 1 << (3 + 2 * n + 1);
        int max_ngrams = 1 << n;
        int ctx_words = (max_possible_contexts + 31) / 32;
        int ngram_words = (max_ngrams + 31) / 32;
        int total_ws_words = ctx_words + 2 * ngram_words;
        size_t shared_mem = THREADS_PER_BLOCK * total_ws_words * sizeof(uint32_t);

        printf("[GPU %d] Using shared memory kernel, shared_mem=%zu bytes (%.1f KB)\n",
               gpu->gpu_id, shared_mem, shared_mem / 1024.0);

        ngramcps_shared_kernel<<<gpu->blocks, gpu->threads, shared_mem, gpu->stream>>>(
            gpu->d_tm_list, gpu->d_results,
            n, max_contexts, max_iterations, timeout_cycles, total_tms,
            gpu->d_processed, gpu->d_nonhalt, gpu->d_unknown
        );
    } else {
        printf("[GPU %d] Using global workspace kernel\n", gpu->gpu_id);

        ngramcps_persistent_kernel<<<gpu->blocks, gpu->threads, 0, gpu->stream>>>(
            gpu->d_tm_list, gpu->d_results,
            n, max_contexts, max_iterations, timeout_cycles, total_tms,
            gpu->d_workspace_pool, gpu->workspace_words_per_thread,
            gpu->d_processed, gpu->d_nonhalt, gpu->d_unknown
        );
    }

    CUDA_CHECK_LAST("kernel launch");
}

void gpu_get_counters(GPUResources* gpu, unsigned long long* processed,
                      unsigned long long* nonhalt, unsigned long long* unknown) {
    CUDA_CHECK(cudaSetDevice(gpu->gpu_id));
    CUDA_CHECK(cudaMemcpy(processed, gpu->d_processed, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nonhalt, gpu->d_nonhalt, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unknown, gpu->d_unknown, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
}

void gpu_get_results(GPUResources* gpu, TMResult* host_results, int count) {
    CUDA_CHECK(cudaSetDevice(gpu->gpu_id));
    CUDA_CHECK(cudaMemcpy(host_results, gpu->d_results, count * sizeof(TMResult),
                          cudaMemcpyDeviceToHost));
}

void gpu_sync(GPUResources* gpu) {
    CUDA_CHECK(cudaSetDevice(gpu->gpu_id));
    CUDA_CHECK(cudaStreamSynchronize(gpu->stream));
}

bool gpu_is_done(GPUResources* gpu) {
    CUDA_CHECK(cudaSetDevice(gpu->gpu_id));
    cudaError_t err = cudaStreamQuery(gpu->stream);
    return (err == cudaSuccess);
}

// Checkpoint and Logging

struct CheckpointState {
    std::mutex mutex;

    std::string base_path;
    std::string nonhalt_path;
    std::string unknown_path;
    std::string log_path;

    std::ofstream nonhalt_file;
    std::ofstream unknown_file;
    std::ofstream log_file;

    std::atomic<unsigned long long> total_processed{0};
    std::atomic<unsigned long long> total_nonhalt{0};
    std::atomic<unsigned long long> total_halt{0};
    std::atomic<unsigned long long> total_unknown{0};

    std::chrono::steady_clock::time_point start_time;
    bool initialized = false;
};

void checkpoint_init(CheckpointState* cp, const std::string& base_path, bool append = false) {
    std::lock_guard<std::mutex> lock(cp->mutex);

    cp->base_path = base_path;
    cp->nonhalt_path = base_path + ".nonhalt.txt";
    cp->unknown_path = base_path + ".unknown.txt";
    cp->log_path = base_path + ".progress.log";

    auto mode = append ? std::ios::app : std::ios::trunc;

    cp->nonhalt_file.open(cp->nonhalt_path, mode);
    cp->unknown_file.open(cp->unknown_path, mode);
    cp->log_file.open(cp->log_path, mode);

    if (!cp->nonhalt_file.is_open()) {
        fprintf(stderr, "Warning: Cannot open %s\n", cp->nonhalt_path.c_str());
    }
    if (!cp->unknown_file.is_open()) {
        fprintf(stderr, "Warning: Cannot open %s\n", cp->unknown_path.c_str());
    }
    if (!cp->log_file.is_open()) {
        fprintf(stderr, "Warning: Cannot open %s\n", cp->log_path.c_str());
    }

    cp->start_time = std::chrono::steady_clock::now();
    cp->initialized = true;

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now_c));
    cp->log_file << "[" << timebuf << "] NGramCPS Decider started\n";
    cp->log_file << "[" << timebuf << "] Base path: " << base_path << "\n";
    cp->log_file.flush();
}

void checkpoint_write_result(CheckpointState* cp, const std::string& seed,
                             const TMResult& result, int round_num, int n) {
    std::lock_guard<std::mutex> lock(cp->mutex);
    if (!cp->initialized) return;

    if (result.result == RESULT_NONHALT && cp->nonhalt_file.is_open()) {
        cp->nonhalt_file << seed << "\tNGramCPS\tn=" << (int)result.n_used
                         << "\tclosure_size=" << result.closure_size
                         << "\tngrams_L=" << result.ngrams_left_count
                         << "\tngrams_R=" << result.ngrams_right_count
                         << "\titerations=" << result.iterations
                         << "\tround=" << round_num << "\n";
        cp->nonhalt_file.flush();
    } else if ((result.result == RESULT_UNKNOWN || result.result == RESULT_HALT)
               && cp->unknown_file.is_open()) {
        const char* reason = (result.result == RESULT_HALT) ? "halt_transition" : "timeout";
        cp->unknown_file << seed << "\t" << reason
                         << "\tclosure_size=" << result.closure_size
                         << "\titerations=" << result.iterations
                         << "\tround=" << round_num << "\tn=" << n << "\n";
        cp->unknown_file.flush();
    }
}

void checkpoint_write_progress(CheckpointState* cp, int round_num, int gpu_id,
                               unsigned long long processed, unsigned long long total,
                               unsigned long long nonhalt, unsigned long long unknown,
                               bool final = false) {
    std::lock_guard<std::mutex> lock(cp->mutex);
    if (!cp->initialized || !cp->log_file.is_open()) return;

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now_c));

    auto elapsed = std::chrono::steady_clock::now() - cp->start_time;
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    double rate = 0.0;
    double eta_sec = 0.0;
    if (elapsed_sec > 0 && processed > 0) {
        rate = (double)processed / elapsed_sec;
        if (processed < total && rate > 0) {
            eta_sec = (double)(total - processed) / rate;
        }
    }

    int pct = total > 0 ? (int)((processed * 100ULL) / total) : 0;

    cp->log_file << "[" << timebuf << "]"
                 << " Round=" << round_num
                 << " GPU=" << gpu_id
                 << " Processed=" << processed << "/" << total
                 << " (" << pct << "%)"
                 << " Rate=" << (int)rate << "TM/s"
                 << " Nonhalt=" << nonhalt
                 << " Unknown=" << unknown
                 << " Elapsed=" << elapsed_sec << "s";
    if (!final && eta_sec > 0) {
        cp->log_file << " ETA=" << (int)eta_sec << "s";
    }
    cp->log_file << "\n";
    cp->log_file.flush();
}

void checkpoint_flush(CheckpointState* cp) {
    std::lock_guard<std::mutex> lock(cp->mutex);
    if (cp->nonhalt_file.is_open()) cp->nonhalt_file.flush();
    if (cp->unknown_file.is_open()) cp->unknown_file.flush();
    if (cp->log_file.is_open()) cp->log_file.flush();
}

void checkpoint_close(CheckpointState* cp) {
    std::lock_guard<std::mutex> lock(cp->mutex);
    if (cp->nonhalt_file.is_open()) cp->nonhalt_file.close();
    if (cp->unknown_file.is_open()) cp->unknown_file.close();
    if (cp->log_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        char timebuf[64];
        strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now_c));
        cp->log_file << "[" << timebuf << "] NGramCPS Decider finished\n";
        cp->log_file.close();
    }
    cp->initialized = false;
}

// Progress Monitor Thread

struct MonitorState {
    std::atomic<bool> should_stop{false};
    std::atomic<bool> is_done{false};

    int round_num;
    int num_gpus;
    unsigned long long total_tms;

    std::vector<GPUResources>* gpus;
    CheckpointState* checkpoint;
};

void progress_monitor_thread(MonitorState* state) {
    auto last_log_time = std::chrono::steady_clock::now();

    while (!state->should_stop.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (state->should_stop.load()) break;

        unsigned long long total_processed = 0;
        unsigned long long total_nonhalt = 0;
        unsigned long long total_unknown = 0;
        bool all_done = true;

        for (int g = 0; g < state->num_gpus; g++) {
            unsigned long long p = 0, nh = 0, uk = 0;
            gpu_get_counters(&(*state->gpus)[g], &p, &nh, &uk);
            total_processed += p;
            total_nonhalt += nh;
            total_unknown += uk;

            if (!gpu_is_done(&(*state->gpus)[g])) {
                all_done = false;
            }
        }

        state->checkpoint->total_processed.store(total_processed);
        state->checkpoint->total_nonhalt.store(total_nonhalt);
        state->checkpoint->total_unknown.store(total_unknown);

        auto now = std::chrono::steady_clock::now();
        auto log_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_log_time).count();

        if (log_elapsed >= PROGRESS_LOG_INTERVAL_SEC || all_done) {
            for (int g = 0; g < state->num_gpus; g++) {
                unsigned long long p = 0, nh = 0, uk = 0;
                gpu_get_counters(&(*state->gpus)[g], &p, &nh, &uk);
                unsigned long long chunk_size = state->total_tms / state->num_gpus;
                unsigned long long gpu_total = (g == state->num_gpus - 1)
                    ? state->total_tms - g * chunk_size
                    : chunk_size;
                checkpoint_write_progress(
                    state->checkpoint, state->round_num, g, p, gpu_total, nh, uk, all_done
                );
            }
            last_log_time = now;
        }

        if (all_done) {
            state->is_done.store(true);
            break;
        }
    }
}

// Round Execution

struct RoundConfig {
    int round_num;
    int n;
    int max_contexts;
    int max_iterations;
    int timeout_per_tm_ms;
    int global_timeout_sec;
    std::string output_base;
};

std::vector<std::string> execute_round(
    const std::vector<std::string>& input_seeds,
    const std::vector<TuringMachine>& input_tms,
    const RoundConfig& config,
    int num_gpus,
    CheckpointState* checkpoint
) {
    printf("\n--- round %d ---\n", config.round_num);
    printf("n=%d, max_contexts=%d, max_iterations=%d, timeout=%dms/TM\n",
           config.n, config.max_contexts, config.max_iterations,
           config.timeout_per_tm_ms);
    printf("Input TMs: %zu\n", input_tms.size());

    if (input_tms.empty()) {
        printf("No TMs to process.\n");
        return {};
    }

    int total_tms = (int)input_tms.size();

    int max_iterations = config.max_iterations;

    std::vector<GPUResources> gpus(num_gpus);
    for (int g = 0; g < num_gpus; g++) {
        init_gpu(&gpus[g], g, config.n, config.max_contexts);
    }

    int tms_per_gpu = (total_tms + num_gpus - 1) / num_gpus;

    std::vector<std::vector<TuringMachine>> tm_chunks(num_gpus);
    std::vector<std::vector<int>> index_chunks(num_gpus);

    for (int g = 0; g < num_gpus; g++) {
        int start = g * tms_per_gpu;
        int end = (start + tms_per_gpu < total_tms) ? start + tms_per_gpu : total_tms;
        for (int i = start; i < end; i++) {
            tm_chunks[g].push_back(input_tms[i]);
            index_chunks[g].push_back(i);
        }
    }

    MonitorState monitor;
    monitor.round_num = config.round_num;
    monitor.num_gpus = num_gpus;
    monitor.total_tms = total_tms;
    monitor.gpus = &gpus;
    monitor.checkpoint = checkpoint;

    std::thread monitor_thread(progress_monitor_thread, &monitor);

    auto launch_start = std::chrono::steady_clock::now();

    for (int g = 0; g < num_gpus; g++) {
        if (!tm_chunks[g].empty()) {
            launch_kernel_on_gpu(
                &gpus[g], tm_chunks[g],
                config.n, config.max_contexts, max_iterations,
                config.timeout_per_tm_ms
            );
        }
    }

    printf("Waiting for GPU kernels to complete...\n");
    for (int g = 0; g < num_gpus; g++) {
        if (!tm_chunks[g].empty()) {
            gpu_sync(&gpus[g]);
        }
    }

    auto launch_end = std::chrono::steady_clock::now();
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
        launch_end - launch_start).count();

    monitor.should_stop.store(true);
    monitor_thread.join();

    printf("Collecting results...\n");
    std::vector<TMResult> all_results(total_tms);
    unsigned long long total_nonhalt = 0;
    unsigned long long total_halt = 0;
    unsigned long long total_unknown = 0;

    for (int g = 0; g < num_gpus; g++) {
        if (tm_chunks[g].empty()) continue;

        int chunk_size = (int)tm_chunks[g].size();
        std::vector<TMResult> chunk_results(chunk_size);
        gpu_get_results(&gpus[g], chunk_results.data(), chunk_size);

        for (int i = 0; i < chunk_size; i++) {
            int global_idx = index_chunks[g][i];
            all_results[global_idx] = chunk_results[i];
        }

        unsigned long long p = 0, nh = 0, uk = 0;
        gpu_get_counters(&gpus[g], &p, &nh, &uk);
        total_nonhalt += nh;
        total_unknown += uk;
        total_halt += 0;
    }

    std::vector<std::string> unknown_seeds;

    for (int i = 0; i < total_tms; i++) {
        const TMResult& res = all_results[i];
        const std::string& seed = input_seeds[i];

        if (res.result == RESULT_NONHALT) {
            checkpoint_write_result(checkpoint, seed, res, config.round_num, config.n);
        } else {
            unknown_seeds.push_back(seed);
            checkpoint_write_result(checkpoint, seed, res, config.round_num, config.n);
        }
    }

    checkpoint_write_progress(
        checkpoint, config.round_num, -1,
        total_tms, total_tms, total_nonhalt, total_unknown, true
    );

    for (int g = 0; g < num_gpus; g++) {
        cleanup_gpu(&gpus[g]);
    }

    printf("\nRound %d complete:\n", config.round_num);
    printf("  Total processed: %d\n", total_tms);
    printf("  Nonhalt: %llu (%.2f%%)\n", total_nonhalt,
           100.0 * total_nonhalt / total_tms);
    printf("  Halt: %llu (%.2f%%)\n", total_halt,
           100.0 * total_halt / total_tms);
    printf("  Unknown/Timeout: %llu (%.2f%%)\n", total_unknown,
           100.0 * total_unknown / total_tms);
    printf("  Time: %lld seconds\n", (long long)elapsed_sec);
    printf("  Rate: %.1f TM/s\n", total_tms / (double)(elapsed_sec > 0 ? elapsed_sec : 1));
    printf("  -> %zu TMs passed to next round\n", unknown_seeds.size());

    checkpoint_flush(checkpoint);

    return unknown_seeds;
}

// I/O Helpers

std::vector<std::string> read_seed_file(const std::string& filepath) {
    std::vector<std::string> seeds;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open input file: %s\n", filepath.c_str());
        return seeds;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        std::string trimmed = line.substr(start, end - start + 1);
        if (trimmed[0] == '#') continue;
        if (trimmed.length() < 10) continue;
        seeds.push_back(trimmed);
    }

    printf("Read %zu seeds from %s\n", seeds.size(), filepath.c_str());
    return seeds;
}

std::pair<std::vector<TuringMachine>, std::vector<std::string>>
parse_seeds(const std::vector<std::string>& seeds) {
    std::vector<TuringMachine> tms;
    std::vector<std::string> normalized;

    tms.reserve(seeds.size());
    normalized.reserve(seeds.size());

    char norm_buf[80];

    for (const auto& seed : seeds) {
        TuringMachine tm;
        if (parse_tm_seed(seed.c_str(), &tm, norm_buf, sizeof(norm_buf))) {
            tms.push_back(tm);
            normalized.push_back(norm_buf);
        } else {
            fprintf(stderr, "Warning: Failed to parse seed: %s\n", seed.c_str());
        }
    }

    printf("Parsed %zu/%zu valid TMs\n", tms.size(), seeds.size());
    return {std::move(tms), std::move(normalized)};
}

void write_unknown_file(const std::string& filepath,
                        const std::vector<std::string>& unknown_seeds) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Warning: Cannot write unknown file: %s\n", filepath.c_str());
        return;
    }

    for (const auto& seed : unknown_seeds) {
        file << seed << "\n";
    }

    printf("Wrote %zu unknown seeds to %s\n", unknown_seeds.size(), filepath.c_str());
}

// Command Line Arguments

struct ProgramOptions {
    std::string input_file;
    std::string output_base = "bb7_ngramcps";

    bool do_round1 = true;
    bool do_round2 = true;
    bool do_round3 = true;

    int timeout_per_tm_ms = 100;
    int global_timeout_sec = 0;

    int num_gpus = 0;

    int r1_n = 2;
    int r1_max_contexts = 10000;
    int r1_max_iterations = 100000;
    int r1_timeout_ms = 10;

    int r2_n = 3;
    int r2_max_contexts = 50000;
    int r2_max_iterations = 500000;
    int r2_timeout_ms = 100;

    int r3_n = 4;
    int r3_max_contexts = 200000;
    int r3_max_iterations = 2000000;
    int r3_timeout_ms = 1000;
};

void print_usage(const char* prog) {
    printf("Usage: %s <input_file> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --output-base <path>       Base path for output files (default: bb7_ngramcps)\n");
    printf("  --round1                   Enable round 1 (default: on)\n");
    printf("  --round2                   Enable round 2 (default: on)\n");
    printf("  --round3                   Enable round 3 (default: on)\n");
    printf("  --no-round1                Disable round 1\n");
    printf("  --no-round2                Disable round 2\n");
    printf("  --no-round3                Disable round 3\n");
    printf("  --timeout-per-tm <ms>      Default timeout per TM in ms (default: 100)\n");
    printf("  --global-timeout <sec>     Global timeout in seconds (default: 0=none)\n");
    printf("  --gpus <N>                 Number of GPUs to use (default: all)\n");
    printf("  --r1-n <N>                 Round 1 n parameter (default: 2)\n");
    printf("  --r1-contexts <N>          Round 1 max contexts (default: 10000)\n");
    printf("  --r1-timeout <ms>          Round 1 timeout ms (default: 10)\n");
    printf("  --r2-n <N>                 Round 2 n parameter (default: 3)\n");
    printf("  --r2-contexts <N>          Round 2 max contexts (default: 50000)\n");
    printf("  --r2-timeout <ms>          Round 2 timeout ms (default: 100)\n");
    printf("  --r3-n <N>                 Round 3 n parameter (default: 4)\n");
    printf("  --r3-contexts <N>          Round 3 max contexts (default: 200000)\n");
    printf("  --r3-timeout <ms>          Round 3 timeout ms (default: 1000)\n");
    printf("  --help                     Show this help\n");
    printf("\nCompile-time tunables:\n");
    printf("  BLOCKS_PER_SM: Override via -DBLOCKS_PER_SM=N (default: %d)\n", BLOCKS_PER_SM_CURRENT);
    printf("    Suggested: CC 5.x-8.x -> 4, CC 8.9 (RTX 40) -> 4, CC 10.x (Blackwell) -> 4-8\n");
    printf("\nSupported GPU series:\n");
    printf("  Consumer: GTX 10/16, RTX 20/30/40/50 series\n");
    printf("  Pro:      RTX Pro 4000/5000/6000\n");
    printf("  Datacenter: A100, H100, H200, B100, B200 (and variants)\n");
    printf("  Requires: Compute Capability >= 5.0, CUDA >= 11.0\n");
    printf("  Note: GPU timeout uses per-device clock rate queried at runtime.\n");
}

ProgramOptions parse_args(int argc, char** argv) {
    ProgramOptions opts;

    if (argc < 2) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    opts.input_file = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "--output-base" && i + 1 < argc) {
            opts.output_base = argv[++i];
        } else if (arg == "--round1") {
            opts.do_round1 = true;
        } else if (arg == "--round2") {
            opts.do_round2 = true;
        } else if (arg == "--round3") {
            opts.do_round3 = true;
        } else if (arg == "--no-round1") {
            opts.do_round1 = false;
        } else if (arg == "--no-round2") {
            opts.do_round2 = false;
        } else if (arg == "--no-round3") {
            opts.do_round3 = false;
        } else if (arg == "--timeout-per-tm" && i + 1 < argc) {
            opts.timeout_per_tm_ms = std::atoi(argv[++i]);
        } else if (arg == "--global-timeout" && i + 1 < argc) {
            opts.global_timeout_sec = std::atoi(argv[++i]);
        } else if (arg == "--gpus" && i + 1 < argc) {
            opts.num_gpus = std::atoi(argv[++i]);
        } else if (arg == "--r1-n" && i + 1 < argc) {
            opts.r1_n = std::atoi(argv[++i]);
        } else if (arg == "--r1-contexts" && i + 1 < argc) {
            opts.r1_max_contexts = std::atoi(argv[++i]);
        } else if (arg == "--r1-timeout" && i + 1 < argc) {
            opts.r1_timeout_ms = std::atoi(argv[++i]);
        } else if (arg == "--r2-n" && i + 1 < argc) {
            opts.r2_n = std::atoi(argv[++i]);
        } else if (arg == "--r2-contexts" && i + 1 < argc) {
            opts.r2_max_contexts = std::atoi(argv[++i]);
        } else if (arg == "--r2-timeout" && i + 1 < argc) {
            opts.r2_timeout_ms = std::atoi(argv[++i]);
        } else if (arg == "--r3-n" && i + 1 < argc) {
            opts.r3_n = std::atoi(argv[++i]);
        } else if (arg == "--r3-contexts" && i + 1 < argc) {
            opts.r3_max_contexts = std::atoi(argv[++i]);
        } else if (arg == "--r3-timeout" && i + 1 < argc) {
            opts.r3_timeout_ms = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return opts;
}

// Signal Handling

static std::atomic<bool> g_interrupt_received{false};

void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down gracefully...\n", sig);
    g_interrupt_received.store(true);
}

// Main

int main(int argc, char** argv) {
    ProgramOptions opts = parse_args(argc, argv);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    printf("BB7 NGramCPS Decider\n");
    printf("Compile-time config: BLOCKS_PER_SM=%d, THREADS_PER_BLOCK=%d\n\n",
           BLOCKS_PER_SM_CURRENT, THREADS_PER_BLOCK);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("CUDA devices detected: %d\n", device_count);

    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA devices found!\n");
        return EXIT_FAILURE;
    }

    int num_gpus = (opts.num_gpus > 0) ? std::min(opts.num_gpus, device_count) : device_count;
    printf("Using %d GPU(s)\n", num_gpus);

    for (int d = 0; d < device_count; d++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        int cr = 0;
        cudaDeviceGetAttribute(&cr, cudaDevAttrClockRate, d);
        if (cr == 0) cr = 2000000;
        printf("  [%d] %s (SMs=%d, CC=%d.%d, Clock=%.0f MHz, Mem=%.1f GB)\n",
               d, prop.name, prop.multiProcessorCount,
               prop.major, prop.minor, cr / 1000.0,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    printf("\nReading input from: %s\n", opts.input_file.c_str());
    std::vector<std::string> seeds = read_seed_file(opts.input_file);

    if (seeds.empty()) {
        fprintf(stderr, "Error: No valid seeds found in input file.\n");
        return EXIT_FAILURE;
    }

    auto [tms, normalized] = parse_seeds(seeds);

    if (tms.empty()) {
        fprintf(stderr, "Error: No valid TMs parsed.\n");
        return EXIT_FAILURE;
    }

    printf("Ready to process %zu TMs\n\n", tms.size());

    CheckpointState checkpoint;
    checkpoint_init(&checkpoint, opts.output_base);

    std::vector<std::string> current_seeds = normalized;
    std::vector<TuringMachine> current_tms = tms;

    if (opts.do_round1 && !g_interrupt_received.load()) {
        RoundConfig r1 = {
            1, opts.r1_n, opts.r1_max_contexts, opts.r1_max_iterations,
            opts.r1_timeout_ms, 0, opts.output_base + "_r1"
        };

        current_seeds = execute_round(current_seeds, current_tms, r1, num_gpus, &checkpoint);

        auto [next_tms, next_normalized] = parse_seeds(current_seeds);
        current_tms = std::move(next_tms);
        current_seeds = std::move(next_normalized);
    }

    if (opts.do_round2 && !current_seeds.empty() && !g_interrupt_received.load()) {
        RoundConfig r2 = {
            2, opts.r2_n, opts.r2_max_contexts, opts.r2_max_iterations,
            opts.r2_timeout_ms, 0, opts.output_base + "_r2"
        };

        current_seeds = execute_round(current_seeds, current_tms, r2, num_gpus, &checkpoint);

        auto [next_tms, next_normalized] = parse_seeds(current_seeds);
        current_tms = std::move(next_tms);
        current_seeds = std::move(next_normalized);
    }

    if (opts.do_round3 && !current_seeds.empty() && !g_interrupt_received.load()) {
        RoundConfig r3 = {
            3, opts.r3_n, opts.r3_max_contexts, opts.r3_max_iterations,
            opts.r3_timeout_ms, 0, opts.output_base + "_r3"
        };

        current_seeds = execute_round(current_seeds, current_tms, r3, num_gpus, &checkpoint);
    }

    std::string final_unknown_path = opts.output_base + ".unknown_final.txt";
    write_unknown_file(final_unknown_path, current_seeds);

    printf("\nFINAL SUMMARY\n");
    printf("Input TMs:          %zu\n", tms.size());
    printf("Final unknown:      %zu (%.4f%%)\n",
           current_seeds.size(),
           100.0 * current_seeds.size() / tms.size());
    printf("Proved NONHALT:     %llu (%.4f%%)\n",
           checkpoint.total_nonhalt.load(),
           100.0 * checkpoint.total_nonhalt.load() / tms.size());
    printf("\nOutput files:\n");
    printf("  Nonhalt: %s\n", checkpoint.nonhalt_path.c_str());
    printf("  Unknown: %s\n", checkpoint.unknown_path.c_str());
    printf("  Log:     %s\n", checkpoint.log_path.c_str());
    printf("  Final:   %s\n", final_unknown_path.c_str());

    checkpoint_close(&checkpoint);

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
