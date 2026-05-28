// gpu_quick_sim_v3.cu -- K=32 Macro Machine Halt Detection on GPU
// Compile: nvcc -O3 -arch=native -std=c++17 -o gpu_quick_sim_v3 gpu_quick_sim_v3.cu
// Usage:   ./gpu_quick_sim_v3 <holdouts_file> <gpu_id> [out_prefix] [max_steps] [timeout_sec] [batch_size] [--low-vram]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Compile-time tunables (override with -DNAME=VALUE)
#ifndef K
#define K           32          // macro block size
#endif

#ifndef TAPE_BLOCKS
#define TAPE_BLOCKS 524288     // 2MB per thread (16M cells / 32)
#endif

#ifndef BATCH_CAP
#define BATCH_CAP   512        // max TMs per batch
#endif

#define HALT_STATE  255
#define MAX_STATES  7
#define MAX_SYMS    2

struct Transition {
    uint8_t w;
    uint8_t m;
    uint8_t ns;
};

struct TM {
    Transition t[MAX_STATES][MAX_SYMS];
};

struct Result {
    uint64_t steps;
    uint8_t  status;  // 0=unknown/max_steps, 1=halt, 2=tape_overflow
};

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Maps compute capability to marketing name.
static const char* gpu_arch_name(int major, int minor)
{
    int cc = major * 10 + minor;
    switch (cc) {
        case 35:  return "Kepler (GTX 600/700 series)";
        case 37:  return "Kepler (Tesla K80)";
        case 50:  return "Maxwell (GTX 750/900 series)";
        case 52:  return "Maxwell (GTX 900 series)";
        case 60:  return "Pascal (GTX 10-series)";
        case 61:  return "Pascal (GTX 10-series)";
        case 62:  return "Pascal (Tegra)";
        case 70:  return "Volta (V100 / Titan V)";
        case 75:  return "Turing (RTX 20-series / GTX 16-series)";
        case 80:  return "Ampere (A100 / RTX A6000)";
        case 86:  return "Ampere (RTX 30-series)";
        case 87:  return "Ampere (Jetson Orin)";
        case 89:  return "Ada Lovelace (RTX 40-series)";
        case 90:  return "Hopper (H100)";
        case 100: return "Blackwell (B100/B200)";
        case 120: return "Blackwell (RTX 50-series)";
        default:  return "Unknown architecture";
    }
}

// Tape in global mem: d_tapes[tid * tape_blocks + block_pos]
// Macro step fires when head is at any edge (offset==0 || offset==K-1).
// Host pre-zeros tape via cudaMemset (faster than per-thread init).
__global__ void quick_sim_kernel(
    const TM* __restrict__ d_tms,
    uint32_t* __restrict__ d_tapes,
    Result*   __restrict__ d_results,
    int       num_tms,
    uint64_t  max_steps,
    unsigned long long timeout_cycles,
    int       tape_blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tms) return;

    const TM* tm = &d_tms[tid];
    uint32_t* tape = d_tapes + (size_t)tid * tape_blocks;

    int block_pos = tape_blocks / 2;
    int offset    = K / 2;
    int state     = 0;
    uint64_t steps = 0;
    uint64_t loop_iter = 0;

    unsigned long long t_start = clock64();

    while (steps < max_steps) {
        // clock64() timeout avoids cudaHostAllocMapped + sync overhead.
        if ((++loop_iter & 1023) == 0 && timeout_cycles > 0 && (clock64() - t_start) > timeout_cycles) {
            d_results[tid].status = 0;
            d_results[tid].steps  = steps;
            return;
        }

        bool at_edge = (offset == 0) || (offset == K - 1);

        if (__builtin_expect(at_edge, 1)) {
            uint32_t pat = tape[block_pos];
            int pos = offset;
            int s = state;
            bool exited = false;

            #pragma unroll
            for (int j = 0; j < K; j++) {
                uint8_t sym = (pat >> pos) & 1u;
                Transition tr = tm->t[s][sym];

                if (__builtin_expect(tr.ns == HALT_STATE, 0)) {
                    d_results[tid].status = 1;
                    d_results[tid].steps  = steps + j + 1;
                    return;
                }

                pat = (pat & ~(1u << pos)) | ((uint32_t)tr.w << pos);
                pos += tr.m ? 1 : -1;
                s = tr.ns;

                if (pos < 0) {
                    tape[block_pos] = pat;
                    block_pos--;
                    offset = K - 1;
                    state = s;
                    steps += j + 1;
                    exited = true;
                    break;
                }
                if (pos >= K) {
                    tape[block_pos] = pat;
                    block_pos++;
                    offset = 0;
                    state = s;
                    steps += j + 1;
                    exited = true;
                    break;
                }
            }

            if (__builtin_expect(!exited, 0)) {
                tape[block_pos] = pat;
                offset = pos;
                state = s;
                steps += K;
            }
        } else {
            uint8_t sym = (tape[block_pos] >> offset) & 1u;
            Transition tr = tm->t[state][sym];

            if (__builtin_expect(tr.ns == HALT_STATE, 0)) {
                d_results[tid].status = 1;
                d_results[tid].steps  = steps + 1;
                return;
            }

            tape[block_pos] = (tape[block_pos] & ~(1u << offset)) | ((uint32_t)tr.w << offset);
            offset += tr.m ? 1 : -1;
            state = tr.ns;
            steps++;

            if (offset < 0) {
                offset = K - 1;
                block_pos--;
            } else if (offset >= K) {
                offset = 0;
                block_pos++;
            }
        }

        if (__builtin_expect(block_pos < 0 || block_pos >= tape_blocks, 0)) {
            d_results[tid].status = 2;
            d_results[tid].steps  = steps;
            return;
        }
    }

    d_results[tid].status = 0;
    d_results[tid].steps  = max_steps;
}

// Seed format: "1RB 1LC  ..." (space-separated 3-char tokens, _ as space)
static bool parse_seed(const char* seed_str, TM* out_tm)
{
    std::memset(out_tm, 0, sizeof(TM));
    for (int s = 0; s < MAX_STATES; s++)
        for (int sym = 0; sym < MAX_SYMS; sym++)
            out_tm->t[s][sym].ns = HALT_STATE;

    char buf[512];
    std::strncpy(buf, seed_str, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char* p = buf; *p; p++)
        if (*p == '_') *p = ' ';

    int state_idx = 0;
    int sym_idx   = 0;
    char* p = buf;

    while (*p && state_idx < MAX_STATES) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        char tok[4] = {0};
        int ti = 0;
        while (*p && *p != ' ' && *p != '\t' && ti < 3)
            tok[ti++] = *p++;
        if (ti < 3) continue;

        uint8_t w  = (tok[0] == '1') ? 1 : 0;
        uint8_t m  = (tok[1] == 'R' || tok[1] == 'r') ? 1 : 0;
        uint8_t ns = HALT_STATE;
        char nc = tok[2];
        /**/ if (nc == 'A' || nc == 'a') ns = 0;
        else if (nc == 'B' || nc == 'b') ns = 1;
        else if (nc == 'C' || nc == 'c') ns = 2;
        else if (nc == 'D' || nc == 'd') ns = 3;
        else if (nc == 'E' || nc == 'e') ns = 4;
        else if (nc == 'F' || nc == 'f') ns = 5;
        else if (nc == 'G' || nc == 'g') ns = 6;
        else if (nc == 'H' || nc == 'h') ns = HALT_STATE;
        else if (nc == 'Z' || nc == 'z') ns = HALT_STATE;
        else if (nc == '-') ns = HALT_STATE;

        out_tm->t[state_idx][sym_idx].w  = w;
        out_tm->t[state_idx][sym_idx].m  = m;
        out_tm->t[state_idx][sym_idx].ns = ns;

        sym_idx++;
        if (sym_idx >= MAX_SYMS) {
            sym_idx = 0;
            state_idx++;
        }
    }
    return true;
}

static void fmt_steps(char* buf, size_t buflen, uint64_t val)
{
    std::snprintf(buf, buflen, "%llu", (unsigned long long)val);
}

int main(int argc, char** argv)
{
    std::printf("gpu_quick_sim K=%d\n\n", K);

    // --low-vram can appear anywhere in arg list
    bool low_vram = false;
    int filtered_argc = 0;
    char** filtered_argv = (char**)std::malloc(argc * sizeof(char*));
    if (!filtered_argv) {
        std::fprintf(stderr, "FATAL: malloc failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--low-vram") == 0) {
            low_vram = true;
        } else {
            filtered_argv[filtered_argc++] = argv[i];
        }
    }

    if (filtered_argc < 3) {
        std::printf("Usage: %s <holdouts_file> <gpu_id> [out_prefix] [max_steps] [timeout_sec] [batch_size] [--low-vram]\n",
                    filtered_argv[0]);
        std::printf("  holdouts_file: path to file with one seed per line\n");
        std::printf("  gpu_id:      GPU index (0-based, as listed by nvidia-smi)\n");
        std::printf("  out_prefix:  output file prefix (default: quicksim)\n");
        std::printf("  max_steps:   max steps to simulate (default: 100000000000000 = 1e14)\n");
        std::printf("  timeout_sec: wall-clock timeout per TM (default: 43200 = 12h)\n");
        std::printf("  batch_size:  0=auto (default), >0=override auto-calculation\n");
        std::printf("  --low-vram:  halve TAPE_BLOCKS for GPUs with limited VRAM\n");
        std::printf("\nCompile-time tunables (pass -DNAME=VALUE to nvcc):\n");
        std::printf("  K            = macro block size (default: 32)\n");
        std::printf("  TAPE_BLOCKS  = blocks per thread (default: 524288 = 2MB)\n");
        std::printf("  BATCH_CAP    = max TMs per batch (default: 512)\n");
        std::printf("\nGPU Selection:\n");
        std::printf("  Use CUDA_VISIBLE_DEVICES=N to control which physical GPU is seen as gpu_id 0.\n");
        std::printf("  Example: CUDA_VISIBLE_DEVICES=2 %s holdouts.txt 0\n", filtered_argv[0]);
        std::printf("\nSupported GPU Series:\n");
        std::printf("  Pascal (GTX 10-series) and newer are supported.\n");
        std::printf("  Recommended: RTX 30-series (Ampere) or newer for best performance.\n");
        std::printf("  Minimum VRAM: ~2GB for small runs, 8GB+ recommended for large batches.\n");
        std::printf("\nOutput files per GPU:\n");
        std::printf("  <prefix>.gpu<N>.halt.txt          -- confirmed halts (seed\\tsteps)\n");
        std::printf("  <prefix>.gpu<N>.unknown.txt       -- reached max_steps or timeout\n");
        std::printf("  <prefix>.gpu<N>.tape_overflow.txt -- tape exceeded allocated size\n");
        std::printf("  <prefix>.gpu<N>.log               -- real-time progress log\n");
        std::free(filtered_argv);
        return EXIT_FAILURE;
    }

    const char* infile         = filtered_argv[1];
    int         gpu_id         = std::atoi(filtered_argv[2]);
    const char* prefix         = (filtered_argc >= 4) ? filtered_argv[3] : "quicksim";
    uint64_t    max_steps      = (filtered_argc >= 5) ? std::strtoull(filtered_argv[4], nullptr, 10) : 100000000000000ULL;
    int         timeout_sec    = (filtered_argc >= 6) ? std::atoi(filtered_argv[5]) : 43200;
    int         batch_override = (filtered_argc >= 7) ? std::atoi(filtered_argv[6]) : 0;

    // Load TMs
    std::vector<std::string> seeds;
    std::vector<TM>          tms;
    {
        FILE* fp = std::fopen(infile, "r");
        if (!fp) { std::fprintf(stderr, "Error: cannot open %s\n", infile); std::free(filtered_argv); return EXIT_FAILURE; }
        char line[512];
        while (std::fgets(line, sizeof(line), fp)) {
            size_t len = std::strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
            if (line[0] == '\0' || line[0] == '#') continue;
            TM tm;
            if (parse_seed(line, &tm)) { seeds.push_back(line); tms.push_back(tm); }
        }
        std::fclose(fp);
    }

    int nt = (int)seeds.size();
    if (nt == 0) { std::fprintf(stderr, "Error: no valid TMs\n"); std::free(filtered_argv); return EXIT_FAILURE; }

    // Setup GPU
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "FATAL: No CUDA-capable GPUs found on this system.\n");
        std::free(filtered_argv);
        return EXIT_FAILURE;
    }
    if (gpu_id < 0 || gpu_id >= device_count) {
        std::fprintf(stderr, "FATAL: gpu_id=%d is out of range. Found %d GPU(s) (valid: 0-%d).\n",
                     gpu_id, device_count, device_count - 1);
        std::fprintf(stderr, "  Use nvidia-smi to list available GPUs.\n");
        std::fprintf(stderr, "  Use CUDA_VISIBLE_DEVICES=N to remap which GPU is seen as gpu_id 0.\n");
        std::free(filtered_argv);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));
    CUDA_CHECK(cudaFree(0));  // force context creation

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    int effective_tape_blocks = TAPE_BLOCKS;
    if (low_vram) {
        effective_tape_blocks = TAPE_BLOCKS / 2;
        if (effective_tape_blocks < 65536)
            effective_tape_blocks = 65536;  // hard floor: 256KB per thread
        std::printf("[GPU%d] --low-vram enabled: TAPE_BLOCKS reduced %d -> %d (%.1fMB/thread)\n",
                    gpu_id, TAPE_BLOCKS, effective_tape_blocks,
                    (effective_tape_blocks * sizeof(uint32_t)) / (1024.0 * 1024.0));
    }

    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, gpu_id);
    if (clock_rate_khz == 0) clock_rate_khz = 2000000;

    std::printf("[GPU%d] Device: %s\n", gpu_id, prop.name);
    std::printf("[GPU%d] Architecture: %s (compute capability %d.%d)\n",
                gpu_id, gpu_arch_name(prop.major, prop.minor), prop.major, prop.minor);
    std::printf("[GPU%d] SMs: %d | Clock: %.0f MHz | Memory: %.1f GB (%.1f GB free)\n",
                gpu_id, prop.multiProcessorCount, clock_rate_khz / 1000.0f,
                total_mem / (1024.0 * 1024.0 * 1024.0),
                free_mem / (1024.0 * 1024.0 * 1024.0));

    if (prop.major < 5) {
        std::fprintf(stderr, "[GPU%d] WARNING: GPU is Kepler/Maxwell (cc %d.%d). Performance may be poor.\n",
                     gpu_id, prop.major, prop.minor);
    }

    size_t mem_per_tm = (size_t)effective_tape_blocks * sizeof(uint32_t) + sizeof(TM) + sizeof(Result) + 256;

    if (mem_per_tm > free_mem) {
        std::fprintf(stderr, "[GPU%d] FATAL: tape per TM (%.1fMB) > free GPU memory (%.1fMB)\n",
                     gpu_id, mem_per_tm / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
        int suggested = (int)((free_mem * 0.70) / sizeof(uint32_t));
        if (suggested < 65536) suggested = 65536;
        std::fprintf(stderr, "  Try: recompile with -DTAPE_BLOCKS=%d or use --low-vram flag\n", suggested);
        std::free(filtered_argv);
        return EXIT_FAILURE;
    }

    // Leave headroom for CUDA runtime -- if batch tapes exceed 90% of free mem, shrink.
    size_t max_tapes_mem = (size_t)BATCH_CAP * effective_tape_blocks * sizeof(uint32_t);
    if (max_tapes_mem > free_mem * 0.90) {
        int old_blocks = effective_tape_blocks;
        effective_tape_blocks = (int)((free_mem * 0.85) / (BATCH_CAP * sizeof(uint32_t)));
        if (effective_tape_blocks < 65536) effective_tape_blocks = 65536;
        mem_per_tm = (size_t)effective_tape_blocks * sizeof(uint32_t) + sizeof(TM) + sizeof(Result) + 256;
        std::printf("[GPU%d] Auto-reduced TAPE_BLOCKS %d -> %d to stay within VRAM limits\n",
                    gpu_id, old_blocks, effective_tape_blocks);
    }

    int batch_size = (int)((free_mem * 0.80) / mem_per_tm);
    if (batch_size < 1) batch_size = 1;
    if (batch_size > BATCH_CAP) batch_size = BATCH_CAP;
    if (batch_size > nt) batch_size = nt;

    // Long timeouts -> smaller batches so one slow TM doesn't stall a whole batch.
    int orig_batch = batch_size;
    if (batch_override > 0) {
        batch_size = batch_override;
        if (batch_size > nt) batch_size = nt;
        size_t needed = (size_t)batch_size * mem_per_tm;
        if (needed > free_mem * 0.95) {
            std::fprintf(stderr, "[GPU%d] WARNING: requested batch=%d needs %.1fGB > free %.1fGB, capping\n",
                         gpu_id, batch_size, needed / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0));
            batch_size = (int)((free_mem * 0.90) / mem_per_tm);
            if (batch_size < 1) batch_size = 1;
        }
    } else {
        if (timeout_sec > 3600 && batch_size > 128) batch_size = 128;
        else if (timeout_sec > 600 && batch_size > 256) batch_size = 256;
        else if (timeout_sec > 60 && batch_size > 512) batch_size = 512;
    }

    std::printf("[GPU%d] TMs: %d | max_steps=%llu | timeout=%ds | tape=%.1fMB/thread (blocks=%d)\n",
                gpu_id, nt, (unsigned long long)max_steps, timeout_sec,
                (effective_tape_blocks * sizeof(uint32_t)) / (1024.0 * 1024.0), effective_tape_blocks);
    std::printf("[GPU%d] Batch: mem_cap=%d | timeout_shrink=%d | final=%d\n\n",
                gpu_id, orig_batch, orig_batch - batch_size, batch_size);

    char halt_name[256], unk_name[256], overflow_name[256], log_name[256];
    std::snprintf(halt_name,     sizeof(halt_name),     "%s.gpu%d.halt.txt", prefix, gpu_id);
    std::snprintf(unk_name,      sizeof(unk_name),      "%s.gpu%d.unknown.txt", prefix, gpu_id);
    std::snprintf(overflow_name, sizeof(overflow_name), "%s.gpu%d.tape_overflow.txt", prefix, gpu_id);
    std::snprintf(log_name,      sizeof(log_name),      "%s.gpu%d.log", prefix, gpu_id);

    FILE* fp_halt     = std::fopen(halt_name, "w");
    FILE* fp_unk      = std::fopen(unk_name, "w");
    FILE* fp_overflow = std::fopen(overflow_name, "w");
    FILE* fp_log      = std::fopen(log_name, "w");

    if (!fp_halt || !fp_unk || !fp_overflow || !fp_log) {
        std::fprintf(stderr, "Error: cannot open output files\n");
        std::free(filtered_argv);
        return EXIT_FAILURE;
    }

    TM*     d_tms;
    Result* d_results;
    uint32_t* d_tapes;
    CUDA_CHECK(cudaMalloc(&d_tms,     batch_size * sizeof(TM)));
    CUDA_CHECK(cudaMalloc(&d_results, batch_size * sizeof(Result)));
    CUDA_CHECK(cudaMalloc(&d_tapes,   (size_t)batch_size * effective_tape_blocks * sizeof(uint32_t)));

    unsigned long long timeout_cycles = 0;
    if (timeout_sec > 0) {
        timeout_cycles = (unsigned long long)timeout_sec * clock_rate_khz * 1000ULL;
    }

    int total_halt = 0, total_unk = 0, total_overflow = 0;
    char step_buf[32];
    cudaEvent_t evt_start, evt_stop;
    CUDA_CHECK(cudaEventCreate(&evt_start));
    CUDA_CHECK(cudaEventCreate(&evt_stop));
    double total_kernel_ms = 0;

    auto t0 = std::chrono::steady_clock::now();
    int num_batches = (nt + batch_size - 1) / batch_size;

    for (int bidx = 0, bs = 0; bs < nt; bidx++, bs += batch_size) {
        int bc = (bs + batch_size <= nt) ? batch_size : (nt - bs);

        // cudaMemset is hardware-accelerated, faster than per-thread init.
        CUDA_CHECK(cudaMemset(d_tapes, 0, (size_t)bc * effective_tape_blocks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_tms, &tms[bs], bc * sizeof(TM), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_results, 0, bc * sizeof(Result)));

        int threads = 256;
        int blocks  = (bc + threads - 1) / threads;

        CUDA_CHECK(cudaEventRecord(evt_start));
        quick_sim_kernel<<<blocks, threads>>>(d_tms, d_tapes, d_results, bc, max_steps, timeout_cycles, effective_tape_blocks);
        CUDA_CHECK(cudaEventRecord(evt_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(evt_stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evt_start, evt_stop));
        total_kernel_ms += ms;

        std::vector<Result> h_results(bc);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, bc * sizeof(Result), cudaMemcpyDeviceToHost));

        int batch_halt = 0, batch_unk = 0, batch_ovf = 0;
        for (int i = 0; i < bc; i++) {
            const Result& r = h_results[i];
            fmt_steps(step_buf, sizeof(step_buf), r.steps);
            const char* seed = seeds[bs + i].c_str();

            if (r.status == 1) {
                std::fprintf(fp_halt, "%s\t%s\n", seed, step_buf);
                total_halt++;
                batch_halt++;
            } else if (r.status == 2) {
                std::fprintf(fp_overflow, "%s\t%s\n", seed, step_buf);
                total_overflow++;
                batch_ovf++;
            } else {
                std::fprintf(fp_unk, "%s\t%s\n", seed, step_buf);
                total_unk++;
                batch_unk++;
            }
        }

        std::fflush(fp_halt);
        std::fflush(fp_unk);
        std::fflush(fp_overflow);

        int done = bs + bc;
        double pct = 100.0 * done / nt;
        std::fprintf(fp_log, "[GPU%d] Batch %d/%d | %d/%d (%.2f%%) | "
                     "H=%d U=%d O=%d | kernel=%.1fms | total_kernel=%.1fms\n",
                     gpu_id, bidx + 1, num_batches, done, nt, pct,
                     batch_halt, batch_unk, batch_ovf, ms, total_kernel_ms);
        std::fflush(fp_log);

        // Print to stdout for tail -f
        std::printf("[GPU%d] %d/%d (%.2f%%) H=%d U=%d O=%d | batch=%.1fms\n",
                    gpu_id, done, nt, pct, batch_halt, batch_unk, batch_ovf, ms);
        std::fflush(stdout);
    }

    auto t1 = std::chrono::steady_clock::now();
    double wall_sec = std::chrono::duration<double>(t1 - t0).count();

    std::fclose(fp_halt);
    std::fclose(fp_unk);
    std::fclose(fp_overflow);
    std::fprintf(fp_log, "[GPU%d] DONE | H=%d U=%d O=%d | wall=%.1fs | kernel=%.1fms | batches=%d\n",
                 gpu_id, total_halt, total_unk, total_overflow, wall_sec, total_kernel_ms, num_batches);
    std::fclose(fp_log);

    CUDA_CHECK(cudaEventDestroy(evt_start));
    CUDA_CHECK(cudaEventDestroy(evt_stop));
    CUDA_CHECK(cudaFree(d_tms));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_tapes));

    std::free(filtered_argv);

    std::printf("\n[GPU%d] COMPLETE | Halt=%d Unknown=%d Overflow=%d\n",
                gpu_id, total_halt, total_unk, total_overflow);
    std::printf("[GPU%d] Wall-clock: %.2f s | Kernel time: %.1f ms | Batches: %d\n",
                gpu_id, wall_sec, total_kernel_ms, num_batches);
    std::printf("[GPU%d] Output: %s.gpu%d.{halt,unknown,tape_overflow,log}.txt\n",
                gpu_id, prefix, gpu_id);

    return EXIT_SUCCESS;
}
