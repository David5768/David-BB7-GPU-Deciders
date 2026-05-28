/* tc_gpu_v3.cu -- Translated Cyclers GPU Decider for BB7
 * Compile: nvcc -O3 -arch=native -std=c++17 -o tc_gpu tc_gpu_v3.cu
 * Usage: ./tc_gpu <holdouts_file> <gpu_id> [tl] [sl] [timeout_ms] [out_prefix]
 */

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                                    \
    cudaError_t cuda_err = (call);                                               \
    if (cuda_err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(cuda_err), #call);        \
        exit(1);                                                                 \
    }                                                                            \
} while (0)

#define TS  5500
#define MR  20
#define HS  255

struct TM { uint8_t w[14], m[14], ns[14]; };

struct Cell {
    long long last_seen;
    uint8_t sym;
    uint8_t seen;
};

struct Record {
    Cell tape[TS];           // 88KB per record, stays in global mem via d_records
    long long time;          // local mem usage: one 88KB tape[] per thread in kernel
    uint8_t state;
    uint8_t read;
    bool minSide;
    int position;
};

struct Result {
    int status;              // 0=unknown  1=halt  2=loop
    int enter_step;
    int period;
    int shift;
};

// tape equivalence check: does past record match current state on same side?
__device__ bool equiv(bool minSide, const Record* past, const Record* curr) {
    int off = 0;
    while (1) {
        int idx = past->position + off;
        if (idx < 0 || idx >= TS) break;
        const Cell& c = curr->tape[idx];
        if (!c.seen || c.last_seen < past->time) break;
        int ci = curr->position + off;
        if (ci < 0 || ci >= TS) return false;
        if (curr->tape[ci].sym != past->tape[idx].sym) return false;
        off += minSide ? 1 : -1;
    }
    return true;
}

__global__ void tc_kernel(const TM* __restrict__ d_tms, int n, long long tl, int sl,
                          unsigned long long timeout_cycles,
                          Record* d_records, int* d_rcounts, Result* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned long long start_clock = clock64();

    Cell tape[TS];
    for (int i = 0; i < TS; i++) { tape[i].last_seen = -1; tape[i].sym = 0; tape[i].seen = 0; }

    Record* recs = d_records + (long long)idx * MR;
    int* rc = d_rcounts + idx;
    *rc = 0;

    int cp = TS / 2, cs = 0;
    long long ct = 0;
    int minPos = cp, maxPos = cp;
    const TM* tm = &d_tms[idx];

    while (cs >= 0 && cs < 7) {
        if ((ct & 1023) == 0 && timeout_cycles > 0) {
            if (clock64() - start_clock > timeout_cycles) {
                d_results[idx].status = 0;
                return;
            }
        }
        if (ct > tl) { d_results[idx].status = 0; return; }
        if (cp < 0 || cp >= TS) { d_results[idx].status = 0; return; }

        uint8_t rd = tape[cp].sym;

        if (cp < minPos || cp > maxPos) {
            bool minSide = (cp < minPos);
            Record r;
            for (int i = 0; i < TS; i++) r.tape[i] = tape[i];
            r.time = ct; r.position = cp; r.state = cs; r.read = rd; r.minSide = minSide;

            for (int i = 0; i < *rc; i++) {
                if (recs[i].minSide == minSide && recs[i].state == cs && recs[i].read == rd
                    && equiv(minSide, &recs[i], &r)) {
                    d_results[idx].status = 2;
                    d_results[idx].enter_step = (int)(recs[i].time + 1);
                    d_results[idx].period = (int)(ct - recs[i].time);
                    d_results[idx].shift = cp - recs[i].position;
                    return;
                }
            }
            if (*rc < MR) { recs[*rc] = r; (*rc)++; }
            if (cp < minPos) minPos = cp;
            if (cp > maxPos) maxPos = cp;
        }
        if (maxPos - minPos > sl) { d_results[idx].status = 0; return; }

        int ti = cs * 2 + rd;
        uint8_t ns = tm->ns[ti];
        if (ns == HS) {
            d_results[idx].status = 1;
            d_results[idx].enter_step = (int)ct;
            return;
        }
        tape[cp].seen = 1; tape[cp].last_seen = ct; tape[cp].sym = tm->w[ti];
        cp += tm->m[ti] ? 1 : -1; cs = ns; ct++;
    }
    d_results[idx].status = 1;
    d_results[idx].enter_step = (int)(ct - 1);
}

TM parse(const std::string& s) {
    TM tm; for (int i = 0; i < 14; i++) tm.ns[i] = HS;
    size_t p = 0; int st = 0;
    while (p < s.size() && st < 7) {
        size_t u = s.find('_', p);
        std::string g = (u == std::string::npos) ? s.substr(p) : s.substr(p, u - p);
        p = (u == std::string::npos) ? s.size() : u + 1;
        for (int sy = 0; sy < 2 && sy * 3 + 2 < (int)g.size(); sy++) {
            std::string t = g.substr(sy * 3, 3); int idx = st * 2 + sy;
            if (t == "---") { tm.w[idx] = 0; tm.m[idx] = 0; }
            else { tm.w[idx] = t[0] - '0'; tm.m[idx] = (t[1] == 'R') ? 1 : 0; int n = t[2] - 'A'; tm.ns[idx] = (n >= 0 && n < 7) ? n : HS; }
        }
        st++;
    }
    return tm;
}

static void ts() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&t);
    printf("[%02d:%02d:%02d] ", tm->tm_hour, tm->tm_min, tm->tm_sec);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <holdouts_file> <gpu_id> [tl] [sl] [timeout_ms] [out_prefix]\n", argv[0]);
        printf("  tl         : step limit (default: 100M)\n");
        printf("  sl         : space limit (default: 2048)\n");
        printf("  timeout_ms : per-TM device-side timeout in ms (default: 0=none)\n");
        printf("\nSupported GPU architectures:\n");
        printf("  Volta (sm_70): V100\n");
        printf("  Turing (sm_75): RTX 20-series, T4\n");
        printf("  Ampere (sm_80/sm_86): RTX 30-series, A100, A6000\n");
        printf("  Ada (sm_89): RTX 40-series\n");
        printf("  Hopper (sm_90): H100\n");
        printf("  Blackwell (sm_100): RTX 50-series, B100, B200\n");
        return 1;
    }

    std::string infile = argv[1];
    int gpu_id = atoi(argv[2]);
    long long tl = (argc >= 4) ? atoll(argv[3]) : 100000000LL;
    int sl = (argc >= 5) ? atoi(argv[4]) : 2048;
    int timeout_ms = (argc >= 6) ? atoi(argv[5]) : 0;
    std::string prefix = (argc >= 7) ? argv[6] : "tc_out";

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (gpu_id < 0 || gpu_id >= device_count) {
        fprintf(stderr, "FATAL: GPU %d is out of range (found %d GPU%s)\n",
                gpu_id, device_count, device_count == 1 ? "" : "s");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));
    CUDA_CHECK(cudaFree(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));

    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, gpu_id);
    if (clock_rate_khz == 0) clock_rate_khz = 2000000; // fallback: 2 GHz

    unsigned long long timeout_cycles = 0;
    if (timeout_ms > 0) {
        timeout_cycles = (unsigned long long)timeout_ms * clock_rate_khz * 1000ULL;
    }

    ts(); printf("[GPU%d] Device: %s (sm_%d%d) | %.2f GB total VRAM\n",
           gpu_id, prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    std::vector<std::string> seeds;
    std::ifstream fin(infile); std::string line;
    while (std::getline(fin, line)) if (!line.empty()) seeds.push_back(line);
    int nt = (int)seeds.size();

    std::vector<TM> tms(nt);
    for (int i = 0; i < nt; i++) tms[i] = parse(seeds[i]);

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t mem_per_tm = (size_t)MR * sizeof(Record);
    int batch_size = (int)((free_mem * 0.85) / mem_per_tm);
    if (batch_size < 1) batch_size = 1;
    if (batch_size > nt) batch_size = nt;

    ts(); printf("[GPU%d] %d TMs | %.1fGB free / %.1fGB total | batch=%d | tl=%lld | sl=%d | timeout=%dms\n",
           gpu_id, nt, free_mem / 1e9, total_mem / 1e9, batch_size, (long long)tl, sl, timeout_ms);

    char dec_name[256], unk_name[256];
    snprintf(dec_name, sizeof(dec_name), "%s.gpu%d.decided.txt", prefix.c_str(), gpu_id);
    snprintf(unk_name, sizeof(unk_name), "%s.gpu%d.unknown.txt", prefix.c_str(), gpu_id);
    FILE* f_dec = fopen(dec_name, "w");
    FILE* f_unk = fopen(unk_name, "w");

    TM* d_tms;
    int* d_rcounts;
    Result* d_results;
    CUDA_CHECK(cudaMalloc(&d_tms, nt * sizeof(TM)));
    CUDA_CHECK(cudaMalloc(&d_rcounts, nt * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, nt * sizeof(Result)));
    CUDA_CHECK(cudaMemcpy(d_tms, tms.data(), nt * sizeof(TM), cudaMemcpyHostToDevice));

    int total_decided = 0, total_unknown = 0;

    for (int bs = 0; bs < nt; ) {
        int bc = (bs + batch_size <= nt) ? batch_size : (nt - bs);

        Record* d_records = nullptr;
        cudaError_t err = cudaMalloc(&d_records, (long long)bc * MR * sizeof(Record));

        while (err == cudaErrorMemoryAllocation && bc > 1) {
            bc /= 2;
            err = cudaMalloc(&d_records, (long long)bc * MR * sizeof(Record));
            ts(); printf("[GPU%d] OOM retry: halved batch to %d\n", gpu_id, bc);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "[GPU%d] FATAL: cudaMalloc failed even with batch=%d: %s\n",
                    gpu_id, bc, cudaGetErrorString(err));
            return 1;
        }

        auto t0 = std::chrono::steady_clock::now();
        ts(); printf("[GPU%d] batch %d-%d (%d TMs)\n", gpu_id, bs, bs + bc - 1, bc);

        int th = 256;
        int bl = (bc + th - 1) / th;
        tc_kernel<<<bl, th>>>(d_tms + bs, bc, tl, sl, timeout_cycles,
                              d_records, d_rcounts + bs, d_results + bs);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_records));

        std::vector<Result> rs(bc);
        CUDA_CHECK(cudaMemcpy(rs.data(), d_results + bs, bc * sizeof(Result), cudaMemcpyDeviceToHost));

        int batch_dec = 0, batch_unk = 0;
        for (int i = 0; i < bc; i++) {
            const Result& r = rs[i];
            if (r.status == 1) {
                fprintf(f_dec, "%s\tHALT\t%d\n", seeds[bs + i].c_str(), r.enter_step);
                batch_dec++;
            } else if (r.status == 2) {
                fprintf(f_dec, "%s\tLOOP\t%d\t%d\t%d\n", seeds[bs + i].c_str(), r.enter_step, r.period, r.shift);
                batch_dec++;
            } else {
                fprintf(f_unk, "%s\n", seeds[bs + i].c_str());
                batch_unk++;
            }
        }
        fflush(f_dec);
        total_decided += batch_dec;
        total_unknown += batch_unk;

        ts(); printf("[GPU%d] batch done %d-%d | decided=%d unknown=%d | total: D=%d U=%d\n",
               gpu_id, bs, bs + bc - 1, batch_dec, batch_unk, total_decided, total_unknown);

        bs += bc;
    }

    fclose(f_dec);
    fclose(f_unk);
    CUDA_CHECK(cudaFree(d_tms));
    CUDA_CHECK(cudaFree(d_rcounts));
    CUDA_CHECK(cudaFree(d_results));

    ts(); printf("[GPU%d] FINISHED | DECIDED=%d UNKNOWN=%d\n", gpu_id, total_decided, total_unknown);
    return 0;
}
