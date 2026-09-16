#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// -----------------------------------------------------------------------------
// Error checking
// -----------------------------------------------------------------------------

inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline const char* cusparse_status_string(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_NOT_SUPPORTED: return "CUSPARSE_STATUS_NOT_SUPPORTED";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
        default: return "CUSPARSE_STATUS_UNKNOWN";
    }
}

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t _err = (call);                                                              \
        if (_err != cudaSuccess) {                                                              \
            std::ostringstream _oss;                                                            \
            _oss << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "                  \
                 << cudaGetErrorString(_err) << " (" << static_cast<int>(_err) << ')';         \
            throw std::runtime_error(_oss.str());                                               \
        }                                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                                      \
    do {                                                                                        \
        cublasStatus_t _status = (call);                                                        \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                                 \
            std::ostringstream _oss;                                                            \
            _oss << "cuBLAS error at " << __FILE__ << ':' << __LINE__ << ": "                \
                 << cublas_status_string(_status);                                              \
            throw std::runtime_error(_oss.str());                                               \
        }                                                                                       \
    } while (0)

#define CUSPARSE_CHECK(call)                                                                    \
    do {                                                                                        \
        cusparseStatus_t _status = (call);                                                      \
        if (_status != CUSPARSE_STATUS_SUCCESS) {                                               \
            std::ostringstream _oss;                                                            \
            _oss << "cuSPARSE error at " << __FILE__ << ':' << __LINE__ << ": "              \
                 << cusparse_status_string(_status);                                            \
            throw std::runtime_error(_oss.str());                                               \
        }                                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// RAII helpers
// -----------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
        }
    }

    void reset() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count) { allocate(count); }
    ~PinnedBuffer() { reset(); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
        }
    }

    void reset() noexcept {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~CudaStream() { if (stream_) cudaStreamDestroy(stream_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { if (event_) cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

inline double event_elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    return static_cast<double>(ms);
}

// -----------------------------------------------------------------------------
// CLI and configuration
// -----------------------------------------------------------------------------

enum class DataKind { Int32, Float32, Float64 };

std::string data_kind_name(DataKind kind) {
    switch (kind) {
        case DataKind::Int32: return "int";
        case DataKind::Float32: return "float";
        case DataKind::Float64: return "double";
    }
    return "unknown";
}

DataKind parse_data_kind(const std::string& text) {
    if (text == "int" || text == "int32") return DataKind::Int32;
    if (text == "float" || text == "fp32") return DataKind::Float32;
    if (text == "double" || text == "fp64") return DataKind::Float64;
    throw std::invalid_argument("Unknown data type: " + text + ". Use int, float, or double.");
}

struct Options {
    int n = 0;
    DataKind kind = DataKind::Float32;
    int iterations = 5;
    int warmup = 2;
    std::uint64_t seed = 12345;
    double zero_probability = 0.0;
    int device = 0;
    int cpu_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int cpu_naive_max_n = 768;
    int cpu_blocked_max_n = 2048;
    int cpu_tile = 64;
    bool verify = true;
    int verify_samples = 64;
    bool allow_tf32 = false;
    bool list_devices = false;
    std::string methods = "all";
    std::string csv_path;
};

void print_usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " N {int|float|double} [options]\n\n"
        << "Required positional arguments:\n"
        << "  N                         Square matrix dimension\n"
        << "  type                      int, float, or double\n\n"
        << "Options:\n"
        << "  --iters K                 Measured iterations (default: 5)\n"
        << "  --warmup K                Warm-up iterations (default: 2)\n"
        << "  --seed S                  Random seed (default: 12345)\n"
        << "  --zero-prob P             Probability that an entry is forced to zero [0,1]\n"
        << "  --device ID               CUDA device id (default: 0)\n"
        << "  --cpu-threads K           Threads for blocked CPU version\n"
        << "  --cpu-naive-max N         Skip CPU naive above N (default: 768)\n"
        << "  --cpu-blocked-max N       Skip CPU blocked above N (default: 2048)\n"
        << "  --cpu-tile K              CPU blocking size (default: 64)\n"
        << "  --verify {0|1}            Verify results (default: 1)\n"
        << "  --verify-samples K        Sample count for large matrices (default: 64)\n"
        << "  --allow-tf32 {0|1}        Let cuBLAS use its default FP32 math mode\n"
        << "  --methods LIST            Comma-separated methods or all\n"
        << "                            cpu_naive,cpu_blocked,cuda_naive,cuda_tiled,cublas,cusparse\n"
        << "  --csv PATH                Append one row per method to CSV\n"
        << "  --list-devices            Print CUDA devices and exit\n"
        << "  --help                    Show this help\n\n"
        << "Example:\n"
        << "  " << program << " 1024 float --iters 10 --csv results/results.csv\n";
}

bool parse_bool(const std::string& value) {
    if (value == "1" || value == "true" || value == "yes") return true;
    if (value == "0" || value == "false" || value == "no") return false;
    throw std::invalid_argument("Expected boolean 0/1, got: " + value);
}

Options parse_options(int argc, char** argv) {
    Options opt;
    if (argc == 2 && std::string(argv[1]) == "--list-devices") {
        opt.list_devices = true;
        return opt;
    }
    if (argc < 3) {
        print_usage(argv[0]);
        throw std::invalid_argument("N and data type are required.");
    }

    opt.n = std::stoi(argv[1]);
    opt.kind = parse_data_kind(argv[2]);
    if (opt.n <= 0) throw std::invalid_argument("N must be positive.");

    for (int i = 3; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + name);
            return argv[++i];
        };

        if (key == "--iters") opt.iterations = std::stoi(need_value(key));
        else if (key == "--warmup") opt.warmup = std::stoi(need_value(key));
        else if (key == "--seed") opt.seed = static_cast<std::uint64_t>(std::stoull(need_value(key)));
        else if (key == "--zero-prob") opt.zero_probability = std::stod(need_value(key));
        else if (key == "--device") opt.device = std::stoi(need_value(key));
        else if (key == "--cpu-threads") opt.cpu_threads = std::stoi(need_value(key));
        else if (key == "--cpu-naive-max") opt.cpu_naive_max_n = std::stoi(need_value(key));
        else if (key == "--cpu-blocked-max") opt.cpu_blocked_max_n = std::stoi(need_value(key));
        else if (key == "--cpu-tile") opt.cpu_tile = std::stoi(need_value(key));
        else if (key == "--verify") opt.verify = parse_bool(need_value(key));
        else if (key == "--verify-samples") opt.verify_samples = std::stoi(need_value(key));
        else if (key == "--allow-tf32") opt.allow_tf32 = parse_bool(need_value(key));
        else if (key == "--methods") opt.methods = need_value(key);
        else if (key == "--csv") opt.csv_path = need_value(key);
        else if (key == "--list-devices") opt.list_devices = true;
        else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + key);
        }
    }

    if (opt.iterations <= 0) throw std::invalid_argument("--iters must be positive.");
    if (opt.warmup < 0) throw std::invalid_argument("--warmup cannot be negative.");
    if (opt.zero_probability < 0.0 || opt.zero_probability > 1.0) {
        throw std::invalid_argument("--zero-prob must be between 0 and 1.");
    }
    if (opt.cpu_threads <= 0) throw std::invalid_argument("--cpu-threads must be positive.");
    if (opt.cpu_tile <= 0) throw std::invalid_argument("--cpu-tile must be positive.");
    if (opt.verify_samples <= 0) throw std::invalid_argument("--verify-samples must be positive.");
    return opt;
}

std::vector<std::string> split_methods(const std::string& text) {
    std::vector<std::string> result;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

bool wants_method(const Options& opt, const std::string& method) {
    if (opt.methods == "all") return true;
    const auto selected = split_methods(opt.methods);
    return std::find(selected.begin(), selected.end(), method) != selected.end();
}

void list_cuda_devices() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        std::cout << "CUDA device query failed: " << cudaGetErrorString(status) << '\n';
        return;
    }
    std::cout << "CUDA devices: " << count << '\n';
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        std::cout << "  [" << i << "] " << p.name
                  << " | CC " << p.major << '.' << p.minor
                  << " | global memory " << std::fixed << std::setprecision(2)
                  << static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << " GiB"
                  << " | SMs " << p.multiProcessorCount << '\n';
    }
}

// -----------------------------------------------------------------------------
// Data generation
// -----------------------------------------------------------------------------

template <typename T>
void fill_random_matrix(std::vector<T>& matrix, std::mt19937_64& rng, double zero_probability) {
    std::bernoulli_distribution force_zero(zero_probability);
    if constexpr (std::is_same_v<T, int>) {
        std::uniform_int_distribution<int> dist(-3, 3);
        for (auto& x : matrix) {
            x = force_zero(rng) ? 0 : dist(rng);
        }
    } else {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& x : matrix) {
            x = force_zero(rng) ? static_cast<T>(0) : static_cast<T>(dist(rng));
        }
    }
}

template <typename T>
std::int64_t count_nonzeros(const std::vector<T>& matrix) {
    return static_cast<std::int64_t>(std::count_if(matrix.begin(), matrix.end(), [](T x) {
        return x != static_cast<T>(0);
    }));
}

// -----------------------------------------------------------------------------
// Native CPU implementations
// -----------------------------------------------------------------------------

template <typename T>
void matmul_cpu_naive(const T* a, const T* b, T* c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if constexpr (std::is_same_v<T, int>) {
                long long sum = 0;
                for (int k = 0; k < n; ++k) {
                    sum += static_cast<long long>(a[static_cast<std::size_t>(i) * n + k]) *
                           static_cast<long long>(b[static_cast<std::size_t>(k) * n + j]);
                }
                c[static_cast<std::size_t>(i) * n + j] = static_cast<int>(sum);
            } else {
                T sum = static_cast<T>(0);
                for (int k = 0; k < n; ++k) {
                    sum += a[static_cast<std::size_t>(i) * n + k] *
                           b[static_cast<std::size_t>(k) * n + j];
                }
                c[static_cast<std::size_t>(i) * n + j] = sum;
            }
        }
    }
}

template <typename T>
void matmul_cpu_blocked_rows(const T* a, const T* b, T* c, int n,
                             int row_begin, int row_end, int tile) {
    for (int ii = row_begin; ii < row_end; ii += tile) {
        const int i_end = std::min(ii + tile, row_end);
        for (int kk = 0; kk < n; kk += tile) {
            const int k_end = std::min(kk + tile, n);
            for (int jj = 0; jj < n; jj += tile) {
                const int j_end = std::min(jj + tile, n);
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        const T aik = a[static_cast<std::size_t>(i) * n + k];
                        for (int j = jj; j < j_end; ++j) {
                            c[static_cast<std::size_t>(i) * n + j] +=
                                aik * b[static_cast<std::size_t>(k) * n + j];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void matmul_cpu_blocked_mt(const T* a, const T* b, T* c, int n, int threads, int tile) {
    std::fill(c, c + static_cast<std::size_t>(n) * n, static_cast<T>(0));
    const int used_threads = std::max(1, std::min(threads, n));
    const int rows_per_thread = (n + used_threads - 1) / used_threads;
    std::vector<std::thread> workers;
    workers.reserve(used_threads);

    for (int t = 0; t < used_threads; ++t) {
        const int row_begin = t * rows_per_thread;
        const int row_end = std::min(n, row_begin + rows_per_thread);
        if (row_begin >= row_end) break;
        workers.emplace_back([=]() {
            matmul_cpu_blocked_rows(a, b, c, n, row_begin, row_end, tile);
        });
    }
    for (auto& worker : workers) worker.join();
}

// -----------------------------------------------------------------------------
// Native CUDA kernels
// -----------------------------------------------------------------------------

template <typename T>
__global__ void matmul_naive_kernel(const T* a, const T* b, T* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    using AccT = std::conditional_t<std::is_same_v<T, int>, long long, T>;
    AccT sum = static_cast<AccT>(0);
    for (int k = 0; k < n; ++k) {
        sum += static_cast<AccT>(a[static_cast<std::size_t>(row) * n + k]) *
               static_cast<AccT>(b[static_cast<std::size_t>(k) * n + col]);
    }
    c[static_cast<std::size_t>(row) * n + col] = static_cast<T>(sum);
}

template <typename T, int TILE>
__global__ void matmul_tiled_kernel(const T* a, const T* b, T* c, int n) {
    __shared__ T tile_a[TILE][TILE];
    __shared__ T tile_b[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    using AccT = std::conditional_t<std::is_same_v<T, int>, long long, T>;
    AccT sum = static_cast<AccT>(0);

    const int tile_count = (n + TILE - 1) / TILE;
    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col = tile * TILE + threadIdx.x;
        const int b_row = tile * TILE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? a[static_cast<std::size_t>(row) * n + a_col] : static_cast<T>(0);
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? b[static_cast<std::size_t>(b_row) * n + col] : static_cast<T>(0);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += static_cast<AccT>(tile_a[threadIdx.y][k]) *
                   static_cast<AccT>(tile_b[k][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = static_cast<T>(sum);
    }
}

// -----------------------------------------------------------------------------
// Statistics and result records
// -----------------------------------------------------------------------------

struct MetricSamples {
    std::vector<double> h2d_ms;
    std::vector<double> compute_ms;
    std::vector<double> d2h_ms;
    std::vector<double> total_gpu_ms;
    std::vector<double> total_wall_ms;
    std::vector<double> launch_overhead_us;
};

struct BasicStats {
    double mean = 0.0;
    double stdev = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

BasicStats stats_of(const std::vector<double>& values) {
    BasicStats s;
    if (values.empty()) return s;
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    const double sq = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    const double variance = std::max(0.0, sq / values.size() - s.mean * s.mean);
    s.stdev = std::sqrt(variance);
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    s.minimum = *min_it;
    s.maximum = *max_it;
    return s;
}

struct VerifyInfo {
    bool checked = false;
    bool passed = true;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    std::size_t checked_elements = 0;
};

struct ResultRow {
    int n = 0;
    std::string data_type;
    std::string method;
    int iterations = 0;
    int warmup = 0;
    int cpu_threads = 0;
    double requested_zero_probability = 0.0;
    std::int64_t nnz_a = 0;
    double actual_sparsity_a = 0.0;
    BasicStats h2d;
    BasicStats compute;
    BasicStats d2h;
    BasicStats total_gpu;
    BasicStats total_wall;
    BasicStats launch_overhead;
    double setup_ms = 0.0;
    double compute_gops = 0.0;
    double end_to_end_gops = 0.0;
    VerifyInfo verify;
    std::string device_name;
    std::string compute_capability;
    std::string notes;
};

template <typename T>
struct TimedOutput {
    std::vector<T> c;
    MetricSamples samples;
    double setup_ms = 0.0;
    std::string notes;
};

// -----------------------------------------------------------------------------
// CPU timing
// -----------------------------------------------------------------------------

template <typename T, typename Function>
TimedOutput<T> benchmark_cpu(const std::vector<T>& a, const std::vector<T>& b,
                             int n, int warmup, int iterations, Function&& function,
                             const std::string& notes = "") {
    TimedOutput<T> out;
    out.c.resize(static_cast<std::size_t>(n) * n);
    out.notes = notes;

    for (int i = 0; i < warmup; ++i) {
        function(a.data(), b.data(), out.c.data());
    }

    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        function(a.data(), b.data(), out.c.data());
        const auto stop = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        out.samples.compute_ms.push_back(ms);
        out.samples.total_wall_ms.push_back(ms);
        out.samples.h2d_ms.push_back(0.0);
        out.samples.d2h_ms.push_back(0.0);
        out.samples.total_gpu_ms.push_back(0.0);
        out.samples.launch_overhead_us.push_back(0.0);
    }
    return out;
}

// -----------------------------------------------------------------------------
// GPU custom-kernel timing
// -----------------------------------------------------------------------------

template <typename T>
TimedOutput<T> benchmark_custom_gpu(const std::vector<T>& a, const std::vector<T>& b,
                                    int n, int warmup, int iterations, bool tiled) {
    TimedOutput<T> out;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const auto setup_start = Clock::now();

    PinnedBuffer<T> h_a(elements), h_b(elements), h_c(elements);
    std::copy(a.begin(), a.end(), h_a.get());
    std::copy(b.begin(), b.end(), h_b.get());
    DeviceBuffer<T> d_a(elements), d_b(elements), d_c(elements);
    CudaStream stream;

    CudaEvent total_start, h2d_start, h2d_end, compute_start, compute_end;
    CudaEvent d2h_start, d2h_end, total_end;

    const auto setup_stop = Clock::now();
    out.setup_ms = std::chrono::duration<double, std::milli>(setup_stop - setup_start).count();

    constexpr int TILE = 16;
    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    auto launch = [&]() -> double {
        const auto host_start = Clock::now();
        if (tiled) {
            matmul_tiled_kernel<T, TILE><<<grid, block, 0, stream.get()>>>(d_a.get(), d_b.get(), d_c.get(), n);
        } else {
            matmul_naive_kernel<T><<<grid, block, 0, stream.get()>>>(d_a.get(), d_b.get(), d_c.get(), n);
        }
        const auto host_stop = Clock::now();
        CUDA_CHECK(cudaPeekAtLastError());
        return std::chrono::duration<double, std::micro>(host_stop - host_start).count();
    };

    auto one_iteration = [&](bool keep) {
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        const auto wall_start = Clock::now();

        CUDA_CHECK(cudaEventRecord(total_start.get(), stream.get()));
        CUDA_CHECK(cudaEventRecord(h2d_start.get(), stream.get()));
        CUDA_CHECK(cudaMemcpyAsync(d_a.get(), h_a.get(), h_a.bytes(), cudaMemcpyHostToDevice, stream.get()));
        CUDA_CHECK(cudaMemcpyAsync(d_b.get(), h_b.get(), h_b.bytes(), cudaMemcpyHostToDevice, stream.get()));
        CUDA_CHECK(cudaEventRecord(h2d_end.get(), stream.get()));

        CUDA_CHECK(cudaEventRecord(compute_start.get(), stream.get()));
        const double launch_us = launch();
        CUDA_CHECK(cudaEventRecord(compute_end.get(), stream.get()));

        CUDA_CHECK(cudaEventRecord(d2h_start.get(), stream.get()));
        CUDA_CHECK(cudaMemcpyAsync(h_c.get(), d_c.get(), h_c.bytes(), cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaEventRecord(d2h_end.get(), stream.get()));
        CUDA_CHECK(cudaEventRecord(total_end.get(), stream.get()));
        CUDA_CHECK(cudaEventSynchronize(total_end.get()));

        const auto wall_stop = Clock::now();
        if (keep) {
            out.samples.h2d_ms.push_back(event_elapsed_ms(h2d_start, h2d_end));
            out.samples.compute_ms.push_back(event_elapsed_ms(compute_start, compute_end));
            out.samples.d2h_ms.push_back(event_elapsed_ms(d2h_start, d2h_end));
            out.samples.total_gpu_ms.push_back(event_elapsed_ms(total_start, total_end));
            out.samples.total_wall_ms.push_back(
                std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
            out.samples.launch_overhead_us.push_back(launch_us);
        }
    };

    for (int i = 0; i < warmup; ++i) one_iteration(false);
    for (int i = 0; i < iterations; ++i) one_iteration(true);

    out.c.assign(h_c.get(), h_c.get() + elements);
    out.notes = tiled
        ? "Native CUDA tiled kernel; 16x16 blocking and shared-memory reuse."
        : "Native CUDA one-thread-per-output-element kernel.";
    return out;
}

// -----------------------------------------------------------------------------
// cuBLAS timing
// -----------------------------------------------------------------------------

inline cublasStatus_t cublas_gemm(cublasHandle_t handle, int n,
                                  const float* a, const float* b, float* c) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C=A*B is computed as column-major C^T=B^T*A^T by swapping A and B.
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, n, n, &alpha, b, n, a, n, &beta, c, n);
}

inline cublasStatus_t cublas_gemm(cublasHandle_t handle, int n,
                                  const double* a, const double* b, double* c) {
    const double alpha = 1.0;
    const double beta = 0.0;
    return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, n, n, &alpha, b, n, a, n, &beta, c, n);
}

template <typename T>
TimedOutput<T> benchmark_cublas_fp(const std::vector<T>& a, const std::vector<T>& b,
                                   int n, int warmup, int iterations, bool allow_tf32) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    TimedOutput<T> out;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const auto setup_start = Clock::now();

    PinnedBuffer<T> h_a(elements), h_b(elements), h_c(elements);
    std::copy(a.begin(), a.end(), h_a.get());
    std::copy(b.begin(), b.end(), h_b.get());
    DeviceBuffer<T> d_a(elements), d_b(elements), d_c(elements);
    CudaStream stream;

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    try {
        CUBLAS_CHECK(cublasSetStream(handle, stream.get()));
        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasSetMathMode(handle,
                allow_tf32 ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH));
        }

        CudaEvent total_start, h2d_start, h2d_end, compute_start, compute_end;
        CudaEvent d2h_start, d2h_end, total_end;

        const auto setup_stop = Clock::now();
        out.setup_ms = std::chrono::duration<double, std::milli>(setup_stop - setup_start).count();

        auto one_iteration = [&](bool keep) {
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));
            const auto wall_start = Clock::now();

            CUDA_CHECK(cudaEventRecord(total_start.get(), stream.get()));
            CUDA_CHECK(cudaEventRecord(h2d_start.get(), stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(d_a.get(), h_a.get(), h_a.bytes(), cudaMemcpyHostToDevice, stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(d_b.get(), h_b.get(), h_b.bytes(), cudaMemcpyHostToDevice, stream.get()));
            CUDA_CHECK(cudaEventRecord(h2d_end.get(), stream.get()));

            CUDA_CHECK(cudaEventRecord(compute_start.get(), stream.get()));
            const auto host_start = Clock::now();
            const cublasStatus_t status = cublas_gemm(handle, n, d_a.get(), d_b.get(), d_c.get());
            const auto host_stop = Clock::now();
            CUBLAS_CHECK(status);
            const double launch_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count();
            CUDA_CHECK(cudaEventRecord(compute_end.get(), stream.get()));

            CUDA_CHECK(cudaEventRecord(d2h_start.get(), stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(h_c.get(), d_c.get(), h_c.bytes(), cudaMemcpyDeviceToHost, stream.get()));
            CUDA_CHECK(cudaEventRecord(d2h_end.get(), stream.get()));
            CUDA_CHECK(cudaEventRecord(total_end.get(), stream.get()));
            CUDA_CHECK(cudaEventSynchronize(total_end.get()));
            const auto wall_stop = Clock::now();

            if (keep) {
                out.samples.h2d_ms.push_back(event_elapsed_ms(h2d_start, h2d_end));
                out.samples.compute_ms.push_back(event_elapsed_ms(compute_start, compute_end));
                out.samples.d2h_ms.push_back(event_elapsed_ms(d2h_start, d2h_end));
                out.samples.total_gpu_ms.push_back(event_elapsed_ms(total_start, total_end));
                out.samples.total_wall_ms.push_back(
                    std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
                out.samples.launch_overhead_us.push_back(launch_us);
            }
        };

        for (int i = 0; i < warmup; ++i) one_iteration(false);
        for (int i = 0; i < iterations; ++i) one_iteration(true);

        out.c.assign(h_c.get(), h_c.get() + elements);
        out.notes = std::is_same_v<T, float>
            ? (allow_tf32
                ? "Native cuBLAS SGEMM; default math mode (TF32 may be selected on supported GPUs)."
                : "Native cuBLAS SGEMM; pedantic FP32 math mode for reproducible verification.")
            : "Native cuBLAS DGEMM.";
    } catch (...) {
        cublasDestroy(handle);
        throw;
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return out;
}

TimedOutput<int> benchmark_cublas_int_emulated(const std::vector<int>& a, const std::vector<int>& b,
                                                int n, int warmup, int iterations) {
    TimedOutput<int> out;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const auto setup_start = Clock::now();

    PinnedBuffer<double> h_a(elements), h_b(elements), h_c(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        h_a.get()[i] = static_cast<double>(a[i]);
        h_b.get()[i] = static_cast<double>(b[i]);
    }
    DeviceBuffer<double> d_a(elements), d_b(elements), d_c(elements);
    CudaStream stream;

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    try {
        CUBLAS_CHECK(cublasSetStream(handle, stream.get()));
        CudaEvent total_start, h2d_start, h2d_end, compute_start, compute_end;
        CudaEvent d2h_start, d2h_end, total_end;

        const auto setup_stop = Clock::now();
        out.setup_ms = std::chrono::duration<double, std::milli>(setup_stop - setup_start).count();

        auto one_iteration = [&](bool keep) {
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));
            const auto wall_start = Clock::now();
            CUDA_CHECK(cudaEventRecord(total_start.get(), stream.get()));
            CUDA_CHECK(cudaEventRecord(h2d_start.get(), stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(d_a.get(), h_a.get(), h_a.bytes(), cudaMemcpyHostToDevice, stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(d_b.get(), h_b.get(), h_b.bytes(), cudaMemcpyHostToDevice, stream.get()));
            CUDA_CHECK(cudaEventRecord(h2d_end.get(), stream.get()));

            CUDA_CHECK(cudaEventRecord(compute_start.get(), stream.get()));
            const auto host_start = Clock::now();
            const cublasStatus_t status = cublas_gemm(handle, n, d_a.get(), d_b.get(), d_c.get());
            const auto host_stop = Clock::now();
            CUBLAS_CHECK(status);
            const double launch_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count();
            CUDA_CHECK(cudaEventRecord(compute_end.get(), stream.get()));

            CUDA_CHECK(cudaEventRecord(d2h_start.get(), stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(h_c.get(), d_c.get(), h_c.bytes(), cudaMemcpyDeviceToHost, stream.get()));
            CUDA_CHECK(cudaEventRecord(d2h_end.get(), stream.get()));
            CUDA_CHECK(cudaEventRecord(total_end.get(), stream.get()));
            CUDA_CHECK(cudaEventSynchronize(total_end.get()));
            const auto wall_stop = Clock::now();

            if (keep) {
                out.samples.h2d_ms.push_back(event_elapsed_ms(h2d_start, h2d_end));
                out.samples.compute_ms.push_back(event_elapsed_ms(compute_start, compute_end));
                out.samples.d2h_ms.push_back(event_elapsed_ms(d2h_start, d2h_end));
                out.samples.total_gpu_ms.push_back(event_elapsed_ms(total_start, total_end));
                out.samples.total_wall_ms.push_back(
                    std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
                out.samples.launch_overhead_us.push_back(launch_us);
            }
        };

        for (int i = 0; i < warmup; ++i) one_iteration(false);
        for (int i = 0; i < iterations; ++i) one_iteration(true);

        out.c.resize(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            out.c[i] = static_cast<int>(std::llround(h_c.get()[i]));
        }
        out.notes = "INT32 emulation through FP64 cuBLAS DGEMM. Exact only while integer sums remain below 2^53; input range is intentionally small.";
    } catch (...) {
        cublasDestroy(handle);
        throw;
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return out;
}

// -----------------------------------------------------------------------------
// cuSPARSE timing
// -----------------------------------------------------------------------------

template <typename T>
struct HostCsr {
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<T> values;
};

template <typename SourceT, typename ValueT>
HostCsr<ValueT> dense_to_csr(const std::vector<SourceT>& a, int n) {
    HostCsr<ValueT> csr;
    csr.row_offsets.resize(static_cast<std::size_t>(n) + 1);
    csr.row_offsets[0] = 0;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const SourceT value = a[static_cast<std::size_t>(row) * n + col];
            if (value != static_cast<SourceT>(0)) {
                if (csr.column_indices.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("CSR nnz exceeds 32-bit index range.");
                }
                csr.column_indices.push_back(col);
                csr.values.push_back(static_cast<ValueT>(value));
            }
        }
        csr.row_offsets[static_cast<std::size_t>(row) + 1] = static_cast<int>(csr.column_indices.size());
    }
    return csr;
}

template <typename T>
constexpr cudaDataType cuda_data_type();

template <>
constexpr cudaDataType cuda_data_type<float>() { return CUDA_R_32F; }

template <>
constexpr cudaDataType cuda_data_type<double>() { return CUDA_R_64F; }

template <typename SourceT, typename ValueT>
TimedOutput<SourceT> benchmark_cusparse_impl(const std::vector<SourceT>& a,
                                             const std::vector<SourceT>& b,
                                             int n, int warmup, int iterations,
                                             const std::string& note) {
    static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>);
    TimedOutput<SourceT> out;
    const std::size_t dense_elements = static_cast<std::size_t>(n) * n;
    const auto setup_start = Clock::now();

    const HostCsr<ValueT> csr = dense_to_csr<SourceT, ValueT>(a, n);
    const std::int64_t nnz = static_cast<std::int64_t>(csr.values.size());

    PinnedBuffer<int> h_row(csr.row_offsets.size());
    PinnedBuffer<int> h_col(std::max<std::size_t>(1, csr.column_indices.size()));
    PinnedBuffer<ValueT> h_values(std::max<std::size_t>(1, csr.values.size()));
    PinnedBuffer<ValueT> h_b(dense_elements), h_c(dense_elements);
    std::copy(csr.row_offsets.begin(), csr.row_offsets.end(), h_row.get());
    if (!csr.column_indices.empty()) std::copy(csr.column_indices.begin(), csr.column_indices.end(), h_col.get());
    if (!csr.values.empty()) std::copy(csr.values.begin(), csr.values.end(), h_values.get());
    for (std::size_t i = 0; i < dense_elements; ++i) h_b.get()[i] = static_cast<ValueT>(b[i]);

    DeviceBuffer<int> d_row(csr.row_offsets.size());
    DeviceBuffer<int> d_col(std::max<std::size_t>(1, csr.column_indices.size()));
    DeviceBuffer<ValueT> d_values(std::max<std::size_t>(1, csr.values.size()));
    DeviceBuffer<ValueT> d_b(dense_elements), d_c(dense_elements);
    CudaStream stream;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c = nullptr;
    void* external_buffer = nullptr;

    CUSPARSE_CHECK(cusparseCreate(&handle));
    try {
        CUSPARSE_CHECK(cusparseSetStream(handle, stream.get()));
        CUSPARSE_CHECK(cusparseCreateCsr(
            &mat_a, n, n, nnz,
            d_row.get(), d_col.get(), d_values.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<ValueT>()));
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &mat_b, n, n, n, d_b.get(), cuda_data_type<ValueT>(), CUSPARSE_ORDER_ROW));
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &mat_c, n, n, n, d_c.get(), cuda_data_type<ValueT>(), CUSPARSE_ORDER_ROW));

        const ValueT alpha = static_cast<ValueT>(1);
        const ValueT beta = static_cast<ValueT>(0);
        std::size_t buffer_size = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_a, mat_b, &beta, mat_c, cuda_data_type<ValueT>(),
            CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
        if (buffer_size > 0) CUDA_CHECK(cudaMalloc(&external_buffer, buffer_size));

        CudaEvent total_start, h2d_start, h2d_end, compute_start, compute_end;
        CudaEvent d2h_start, d2h_end, total_end;

        const auto setup_stop = Clock::now();
        out.setup_ms = std::chrono::duration<double, std::milli>(setup_stop - setup_start).count();

        auto one_iteration = [&](bool keep) {
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));
            const auto wall_start = Clock::now();
            CUDA_CHECK(cudaEventRecord(total_start.get(), stream.get()));
            CUDA_CHECK(cudaEventRecord(h2d_start.get(), stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(d_row.get(), h_row.get(), h_row.bytes(), cudaMemcpyHostToDevice, stream.get()));
            if (nnz > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_col.get(), h_col.get(), static_cast<std::size_t>(nnz) * sizeof(int),
                                           cudaMemcpyHostToDevice, stream.get()));
                CUDA_CHECK(cudaMemcpyAsync(d_values.get(), h_values.get(), static_cast<std::size_t>(nnz) * sizeof(ValueT),
                                           cudaMemcpyHostToDevice, stream.get()));
            }
            CUDA_CHECK(cudaMemcpyAsync(d_b.get(), h_b.get(), h_b.bytes(), cudaMemcpyHostToDevice, stream.get()));
            CUDA_CHECK(cudaEventRecord(h2d_end.get(), stream.get()));

            CUDA_CHECK(cudaEventRecord(compute_start.get(), stream.get()));
            const auto host_start = Clock::now();
            const cusparseStatus_t status = cusparseSpMM(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, mat_a, mat_b, &beta, mat_c, cuda_data_type<ValueT>(),
                CUSPARSE_SPMM_ALG_DEFAULT, external_buffer);
            const auto host_stop = Clock::now();
            CUSPARSE_CHECK(status);
            const double launch_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count();
            CUDA_CHECK(cudaEventRecord(compute_end.get(), stream.get()));

            CUDA_CHECK(cudaEventRecord(d2h_start.get(), stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(h_c.get(), d_c.get(), h_c.bytes(), cudaMemcpyDeviceToHost, stream.get()));
            CUDA_CHECK(cudaEventRecord(d2h_end.get(), stream.get()));
            CUDA_CHECK(cudaEventRecord(total_end.get(), stream.get()));
            CUDA_CHECK(cudaEventSynchronize(total_end.get()));
            const auto wall_stop = Clock::now();

            if (keep) {
                out.samples.h2d_ms.push_back(event_elapsed_ms(h2d_start, h2d_end));
                out.samples.compute_ms.push_back(event_elapsed_ms(compute_start, compute_end));
                out.samples.d2h_ms.push_back(event_elapsed_ms(d2h_start, d2h_end));
                out.samples.total_gpu_ms.push_back(event_elapsed_ms(total_start, total_end));
                out.samples.total_wall_ms.push_back(
                    std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
                out.samples.launch_overhead_us.push_back(launch_us);
            }
        };

        for (int i = 0; i < warmup; ++i) one_iteration(false);
        for (int i = 0; i < iterations; ++i) one_iteration(true);

        out.c.resize(dense_elements);
        for (std::size_t i = 0; i < dense_elements; ++i) {
            if constexpr (std::is_same_v<SourceT, int>) {
                out.c[i] = static_cast<int>(std::llround(h_c.get()[i]));
            } else {
                out.c[i] = static_cast<SourceT>(h_c.get()[i]);
            }
        }
        out.notes = note + " Host dense-to-CSR conversion is reported in setup_ms, not in CUDA-event total time.";

        if (external_buffer) CUDA_CHECK(cudaFree(external_buffer));
        external_buffer = nullptr;
        CUSPARSE_CHECK(cusparseDestroyDnMat(mat_c)); mat_c = nullptr;
        CUSPARSE_CHECK(cusparseDestroyDnMat(mat_b)); mat_b = nullptr;
        CUSPARSE_CHECK(cusparseDestroySpMat(mat_a)); mat_a = nullptr;
    } catch (...) {
        if (external_buffer) cudaFree(external_buffer);
        if (mat_c) cusparseDestroyDnMat(mat_c);
        if (mat_b) cusparseDestroyDnMat(mat_b);
        if (mat_a) cusparseDestroySpMat(mat_a);
        cusparseDestroy(handle);
        throw;
    }
    CUSPARSE_CHECK(cusparseDestroy(handle));
    return out;
}

// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------

template <typename T>
long double reference_entry(const std::vector<T>& a, const std::vector<T>& b,
                            int n, int row, int col) {
    long double sum = 0.0L;
    for (int k = 0; k < n; ++k) {
        sum += static_cast<long double>(a[static_cast<std::size_t>(row) * n + k]) *
               static_cast<long double>(b[static_cast<std::size_t>(k) * n + col]);
    }
    return sum;
}

template <typename T>
VerifyInfo verify_output(const std::vector<T>& a, const std::vector<T>& b,
                         const std::vector<T>& c, int n,
                         const std::vector<T>* full_reference,
                         int sample_count, std::uint64_t seed,
                         bool allow_tf32) {
    VerifyInfo info;
    info.checked = true;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    if (c.size() != elements) {
        info.passed = false;
        return info;
    }

    auto evaluate = [&](std::size_t index, long double expected) {
        const long double actual = static_cast<long double>(c[index]);
        const long double abs_error = std::fabs(actual - expected);
        const long double rel_error = abs_error / std::max<long double>(1.0L, std::fabs(expected));
        info.max_abs_error = std::max(info.max_abs_error, static_cast<double>(abs_error));
        info.max_rel_error = std::max(info.max_rel_error, static_cast<double>(rel_error));
        ++info.checked_elements;

        if constexpr (std::is_same_v<T, int>) {
            if (actual != expected) info.passed = false;
        } else if constexpr (std::is_same_v<T, float>) {
            const long double base = allow_tf32 ? 1.0e-2L : 7.5e-4L;
            const long double atol = base * std::sqrt(static_cast<long double>(n)) + 1.0e-5L;
            const long double rtol = allow_tf32 ? 1.0e-2L : 1.0e-3L;
            if (abs_error > atol + rtol * std::fabs(expected)) info.passed = false;
        } else {
            const long double atol = 2.0e-10L * std::sqrt(static_cast<long double>(n)) + 1.0e-12L;
            const long double rtol = 2.0e-10L;
            if (abs_error > atol + rtol * std::fabs(expected)) info.passed = false;
        }
    };

    if (full_reference && full_reference->size() == elements) {
        for (std::size_t i = 0; i < elements; ++i) {
            evaluate(i, static_cast<long double>((*full_reference)[i]));
        }
    } else {
        std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
        std::uniform_int_distribution<int> dist(0, n - 1);
        const int checks = std::min<int>(sample_count, n * n);
        for (int s = 0; s < checks; ++s) {
            const int row = dist(rng);
            const int col = dist(rng);
            const std::size_t index = static_cast<std::size_t>(row) * n + col;
            evaluate(index, reference_entry(a, b, n, row, col));
        }
    }
    return info;
}

// -----------------------------------------------------------------------------
// Result construction and CSV
// -----------------------------------------------------------------------------

ResultRow make_result(const Options& opt, const std::string& method,
                      const MetricSamples& samples, double setup_ms,
                      const VerifyInfo& verify, std::int64_t nnz,
                      const std::string& device_name,
                      const std::string& compute_capability,
                      const std::string& notes) {
    ResultRow r;
    r.n = opt.n;
    r.data_type = data_kind_name(opt.kind);
    r.method = method;
    r.iterations = opt.iterations;
    r.warmup = opt.warmup;
    r.cpu_threads = opt.cpu_threads;
    r.requested_zero_probability = opt.zero_probability;
    r.nnz_a = nnz;
    const double total_entries = static_cast<double>(opt.n) * opt.n;
    r.actual_sparsity_a = total_entries > 0.0 ? 1.0 - static_cast<double>(nnz) / total_entries : 0.0;
    r.h2d = stats_of(samples.h2d_ms);
    r.compute = stats_of(samples.compute_ms);
    r.d2h = stats_of(samples.d2h_ms);
    r.total_gpu = stats_of(samples.total_gpu_ms);
    r.total_wall = stats_of(samples.total_wall_ms);
    r.launch_overhead = stats_of(samples.launch_overhead_us);
    r.setup_ms = setup_ms;
    r.verify = verify;
    r.device_name = device_name;
    r.compute_capability = compute_capability;
    r.notes = notes;

    const long double operations = 2.0L * opt.n * opt.n * opt.n;
    if (r.compute.mean > 0.0) r.compute_gops = static_cast<double>(operations / (r.compute.mean * 1.0e6L));
    const double end_to_end_ms = r.total_gpu.mean > 0.0 ? r.total_gpu.mean : r.total_wall.mean;
    if (end_to_end_ms > 0.0) r.end_to_end_gops = static_cast<double>(operations / (end_to_end_ms * 1.0e6L));
    return r;
}

std::string csv_escape(const std::string& text) {
    if (text.find_first_of(",\"\n") == std::string::npos) return text;
    std::string escaped = "\"";
    for (char c : text) {
        if (c == '\"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += '\"';
    return escaped;
}

void append_csv(const std::string& path, const std::vector<ResultRow>& rows) {
    if (path.empty() || rows.empty()) return;
    const fs::path csv_path(path);
    if (csv_path.has_parent_path()) fs::create_directories(csv_path.parent_path());
    const bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    std::ofstream file(path, std::ios::app);
    if (!file) throw std::runtime_error("Cannot open CSV file: " + path);

    if (write_header) {
        file << "N,data_type,method,iterations,warmup,cpu_threads,requested_zero_probability,"
             << "nnz_a,actual_sparsity_a,h2d_ms,h2d_stdev_ms,compute_ms,compute_stdev_ms,"
             << "d2h_ms,d2h_stdev_ms,total_gpu_ms,total_gpu_stdev_ms,total_wall_ms,"
             << "total_wall_stdev_ms,launch_overhead_us,launch_overhead_stdev_us,setup_ms,"
             << "compute_gops,end_to_end_gops,verified,max_abs_error,max_rel_error,"
             << "checked_elements,device_name,compute_capability,notes\n";
    }

    file << std::setprecision(12);
    for (const auto& r : rows) {
        file << r.n << ',' << r.data_type << ',' << r.method << ','
             << r.iterations << ',' << r.warmup << ',' << r.cpu_threads << ','
             << r.requested_zero_probability << ',' << r.nnz_a << ',' << r.actual_sparsity_a << ','
             << r.h2d.mean << ',' << r.h2d.stdev << ','
             << r.compute.mean << ',' << r.compute.stdev << ','
             << r.d2h.mean << ',' << r.d2h.stdev << ','
             << r.total_gpu.mean << ',' << r.total_gpu.stdev << ','
             << r.total_wall.mean << ',' << r.total_wall.stdev << ','
             << r.launch_overhead.mean << ',' << r.launch_overhead.stdev << ','
             << r.setup_ms << ',' << r.compute_gops << ',' << r.end_to_end_gops << ','
             << (r.verify.checked ? (r.verify.passed ? "1" : "0") : "") << ','
             << r.verify.max_abs_error << ',' << r.verify.max_rel_error << ','
             << r.verify.checked_elements << ',' << csv_escape(r.device_name) << ','
             << csv_escape(r.compute_capability) << ',' << csv_escape(r.notes) << '\n';
    }
}

void print_results(const std::vector<ResultRow>& rows) {
    std::cout << "\nTiming summary (mean of measured iterations)\n";
    std::cout << std::left
              << std::setw(25) << "method"
              << std::right
              << std::setw(12) << "H2D ms"
              << std::setw(14) << "compute ms"
              << std::setw(12) << "D2H ms"
              << std::setw(14) << "total ms"
              << std::setw(14) << "launch us"
              << std::setw(14) << "GOp/s"
              << std::setw(12) << "verify" << '\n';
    std::cout << std::string(117, '-') << '\n';
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& r : rows) {
        const double total = r.total_gpu.mean > 0.0 ? r.total_gpu.mean : r.total_wall.mean;
        std::cout << std::left << std::setw(25) << r.method << std::right
                  << std::setw(12) << r.h2d.mean
                  << std::setw(14) << r.compute.mean
                  << std::setw(12) << r.d2h.mean
                  << std::setw(14) << total
                  << std::setw(14) << r.launch_overhead.mean
                  << std::setw(14) << r.end_to_end_gops
                  << std::setw(12) << (r.verify.checked ? (r.verify.passed ? "PASS" : "FAIL") : "N/A")
                  << '\n';
    }
    std::cout << '\n';
}

// -----------------------------------------------------------------------------
// Typed benchmark driver
// -----------------------------------------------------------------------------

template <typename T>
int run_typed(const Options& opt, const cudaDeviceProp* prop) {
    const std::size_t elements = static_cast<std::size_t>(opt.n) * opt.n;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("Matrix allocation size overflow.");
    }

    std::vector<T> a(elements), b(elements);
    std::mt19937_64 rng_a(opt.seed);
    std::mt19937_64 rng_b(opt.seed ^ 0xd1b54a32d192ed03ULL);
    fill_random_matrix(a, rng_a, opt.zero_probability);
    fill_random_matrix(b, rng_b, opt.zero_probability);
    const std::int64_t nnz = count_nonzeros(a);

    std::string device_name = prop ? prop->name : "CPU only";
    std::string capability = prop ? (std::to_string(prop->major) + "." + std::to_string(prop->minor)) : "N/A";

    std::cout << "N=" << opt.n << ", type=" << data_kind_name(opt.kind)
              << ", entries per matrix=" << elements
              << ", A sparsity=" << std::fixed << std::setprecision(3)
              << (1.0 - static_cast<double>(nnz) / static_cast<double>(elements)) * 100.0 << "%\n";
    if (prop) std::cout << "GPU: " << device_name << " (CC " << capability << ")\n";

    std::vector<ResultRow> results;
    std::vector<T> reference;
    bool have_full_reference = false;

    auto add_result = [&](const std::string& method, TimedOutput<T>&& timed, bool tf32_for_verify = false) {
        VerifyInfo verify;
        if (opt.verify) {
            verify = verify_output(a, b, timed.c, opt.n,
                                   have_full_reference ? &reference : nullptr,
                                   opt.verify_samples, opt.seed, tf32_for_verify);
        }
        results.push_back(make_result(opt, method, timed.samples, timed.setup_ms, verify, nnz,
                                      device_name, capability, timed.notes));
        if (opt.verify && !verify.passed) {
            std::cerr << "Verification failed for " << method
                      << ": max_abs=" << verify.max_abs_error
                      << ", max_rel=" << verify.max_rel_error << '\n';
        }
    };

    if (wants_method(opt, "cpu_naive")) {
        if (opt.n <= opt.cpu_naive_max_n) {
            auto timed = benchmark_cpu<T>(
                a, b, opt.n, std::min(1, opt.warmup), opt.iterations,
                [&](const T* pa, const T* pb, T* pc) { matmul_cpu_naive(pa, pb, pc, opt.n); },
                "Native scalar triple-loop CPU implementation.");
            if (!have_full_reference) {
                reference = timed.c;
                have_full_reference = true;
            }
            add_result("cpu_naive", std::move(timed));
        } else {
            std::cout << "Skipping cpu_naive because N > --cpu-naive-max.\n";
        }
    }

    if (wants_method(opt, "cpu_blocked")) {
        if (opt.n <= opt.cpu_blocked_max_n) {
            auto timed = benchmark_cpu<T>(
                a, b, opt.n, std::min(1, opt.warmup), opt.iterations,
                [&](const T* pa, const T* pb, T* pc) {
                    matmul_cpu_blocked_mt(pa, pb, pc, opt.n, opt.cpu_threads, opt.cpu_tile);
                },
                "Native blocked multi-threaded CPU implementation using std::thread; no BLAS library.");
            if (!have_full_reference) {
                reference = timed.c;
                have_full_reference = true;
            }
            add_result("cpu_blocked_mt", std::move(timed));
        } else {
            std::cout << "Skipping cpu_blocked because N > --cpu-blocked-max.\n";
        }
    }

    const bool gpu_requested = wants_method(opt, "cuda_naive") || wants_method(opt, "cuda_tiled") ||
                               wants_method(opt, "cublas") || wants_method(opt, "cusparse");
    if (gpu_requested && !prop) throw std::runtime_error("GPU method requested, but no CUDA device is available.");

    if (wants_method(opt, "cuda_naive")) {
        auto timed = benchmark_custom_gpu(a, b, opt.n, opt.warmup, opt.iterations, false);
        add_result("cuda_naive", std::move(timed));
    }
    if (wants_method(opt, "cuda_tiled")) {
        auto timed = benchmark_custom_gpu(a, b, opt.n, opt.warmup, opt.iterations, true);
        add_result("cuda_tiled_shared", std::move(timed));
    }
    if (wants_method(opt, "cublas")) {
        if constexpr (std::is_same_v<T, int>) {
            auto timed = benchmark_cublas_int_emulated(a, b, opt.n, opt.warmup, opt.iterations);
            add_result("cublas_int_via_fp64", std::move(timed));
        } else {
            auto timed = benchmark_cublas_fp(a, b, opt.n, opt.warmup, opt.iterations, opt.allow_tf32);
            add_result(std::is_same_v<T, float> ? "cublas_sgemm" : "cublas_dgemm",
                       std::move(timed), std::is_same_v<T, float> && opt.allow_tf32);
        }
    }
    if (wants_method(opt, "cusparse")) {
        if constexpr (std::is_same_v<T, int>) {
            auto timed = benchmark_cusparse_impl<int, double>(
                a, b, opt.n, opt.warmup, opt.iterations,
                "INT32 emulation through FP64 cuSPARSE SpMM; integer values are converted to double.");
            add_result("cusparse_int_via_fp64", std::move(timed));
        } else if constexpr (std::is_same_v<T, float>) {
            auto timed = benchmark_cusparse_impl<float, float>(
                a, b, opt.n, opt.warmup, opt.iterations,
                "Native cuSPARSE CSR-by-dense SpMM in FP32.");
            add_result("cusparse_spmm_fp32", std::move(timed));
        } else {
            auto timed = benchmark_cusparse_impl<double, double>(
                a, b, opt.n, opt.warmup, opt.iterations,
                "Native cuSPARSE CSR-by-dense SpMM in FP64.");
            add_result("cusparse_spmm_fp64", std::move(timed));
        }
    }

    print_results(results);
    append_csv(opt.csv_path, results);

    const bool all_verified = std::all_of(results.begin(), results.end(), [](const ResultRow& r) {
        return !r.verify.checked || r.verify.passed;
    });
    return all_verified ? 0 : 2;
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_options(argc, argv);
        if (opt.list_devices) {
            list_cuda_devices();
            return 0;
        }

        int device_count = 0;
        cudaError_t count_status = cudaGetDeviceCount(&device_count);
        const bool has_gpu = count_status == cudaSuccess && device_count > 0;
        cudaDeviceProp prop{};
        cudaDeviceProp* prop_ptr = nullptr;

        if (has_gpu) {
            if (opt.device < 0 || opt.device >= device_count) {
                throw std::invalid_argument("Invalid CUDA device id.");
            }
            CUDA_CHECK(cudaSetDevice(opt.device));
            CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
            prop_ptr = &prop;
        }

        switch (opt.kind) {
            case DataKind::Int32: return run_typed<int>(opt, prop_ptr);
            case DataKind::Float32: return run_typed<float>(opt, prop_ptr);
            case DataKind::Float64: return run_typed<double>(opt, prop_ptr);
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
    return 1;
}
