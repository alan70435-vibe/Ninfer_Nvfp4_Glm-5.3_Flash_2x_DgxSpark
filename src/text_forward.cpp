#include "ninfer_glm53/text_forward.hpp"

#include "json_scan.hpp"
#include "ninfer_glm53/dflash_loop.hpp"
#include "ninfer_glm53/model_spec.hpp"
#include "ninfer_glm53/nvfp4_decode.hpp"
#include "ninfer_glm53/speculative.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ninfer::glm53 {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

float sigmoid(float x) {
    if (x >= 0.f) {
        const float z = std::exp(-x);
        return 1.f / (1.f + z);
    }
    const float z = std::exp(x);
    return z / (1.f + z);
}

float silu(float x) { return x * sigmoid(x); }

float bf16_to_f32(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
    float out = 0.f;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

std::uint8_t f32_to_e4m3(float value) {
    if (std::isnan(value)) return 0x7f;
    const bool negative = std::signbit(value);
    float magnitude = std::fabs(value);
    if (magnitude == 0.f) return negative ? 0x80 : 0x00;
    if (magnitude > 448.f) magnitude = 448.f;
    int exponent = 0;
    const float fraction = std::frexp(magnitude, &exponent);
    float normal = fraction * 2.f;
    int biased = exponent - 1 + 7;
    if (biased <= 0) {
        int mantissa = static_cast<int>(std::nearbyint(std::ldexp(magnitude, 9)));
        if (mantissa > 7) mantissa = 7;
        if (mantissa < 0) mantissa = 0;
        return static_cast<std::uint8_t>((negative ? 0x80 : 0) | mantissa);
    }
    if (biased > 15) return static_cast<std::uint8_t>((negative ? 0x80 : 0) | 0x7e);
    int mantissa = static_cast<int>(std::nearbyint((normal - 1.f) * 8.f));
    if (mantissa == 8) {
        mantissa = 0;
        ++biased;
        if (biased > 15) return static_cast<std::uint8_t>((negative ? 0x80 : 0) | 0x7e);
    }
    if (biased >= 15 && mantissa > 6) return static_cast<std::uint8_t>((negative ? 0x80 : 0) | 0x7e);
    return static_cast<std::uint8_t>((negative ? 0x80 : 0) | (biased << 3) | mantissa);
}

std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1U;
    const std::uint32_t round = 0x7fffU + lsb;
    if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>(bits >> 16);
    bits += round;
    return static_cast<std::uint16_t>(bits >> 16);
}

constexpr std::chrono::seconds kRankIoBudget{30};

[[nodiscard]] bool wait_fd(int fd, int events, std::chrono::steady_clock::time_point deadline) {
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return false;
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (millis <= 0) return false;
        const int timeout = millis > 30000 ? 30000 : static_cast<int>(millis);
        pollfd ready {};
        ready.fd = fd;
        ready.events = static_cast<short>(events);
        const int waited = ::poll(&ready, 1, timeout);
        if (waited < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (waited == 0) continue;
        if ((ready.revents & (POLLERR | POLLNVAL)) != 0) return false;
        if ((ready.revents & events) != 0 || (ready.revents & POLLHUP) != 0) return true;
        return false;
    }
}

void send_all(int fd, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const char*>(data);
    std::size_t done = 0;
    const auto deadline = std::chrono::steady_clock::now() + kRankIoBudget;
    while (done < bytes) {
        if (!wait_fd(fd, POLLOUT, deadline)) fail("rank link send failed");
        const ssize_t wrote = ::send(fd, cursor + done, bytes - done, MSG_NOSIGNAL);
        if (wrote < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            fail("rank link send failed");
        }
        if (wrote == 0) fail("rank link send failed");
        done += static_cast<std::size_t>(wrote);
    }
}

void recv_all(int fd, void* data, std::size_t bytes) {
    auto* cursor = static_cast<char*>(data);
    std::size_t done = 0;
    const auto deadline = std::chrono::steady_clock::now() + kRankIoBudget;
    while (done < bytes) {
        if (!wait_fd(fd, POLLIN, deadline)) fail("rank link recv failed");
        const ssize_t got = ::recv(fd, cursor + done, bytes - done, 0);
        if (got < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            fail("rank link recv failed");
        }
        if (got == 0) fail("rank link recv failed");
        done += static_cast<std::size_t>(got);
    }
}

void set_nonblock(int fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0) fail("rank link nonblock failed");
    if (::fcntl(fd, F_SETFL, static_cast<int>(flags | O_NONBLOCK)) != 0) fail("rank link nonblock failed");
}

class SocketFd {
public:
    SocketFd() = default;
    explicit SocketFd(int fd) noexcept : fd_(fd) {}
    SocketFd(const SocketFd&) = delete;
    SocketFd& operator=(const SocketFd&) = delete;
    SocketFd(SocketFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    SocketFd& operator=(SocketFd&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }
    ~SocketFd() { reset(); }

    [[nodiscard]] int get() const { return fd_; }

    void reset() noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

private:
    int fd_ = -1;
};

class ChildProcess {
public:
    ChildProcess() = default;
    explicit ChildProcess(pid_t pid) noexcept : pid_(pid) {}
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    ChildProcess(ChildProcess&& other) noexcept : pid_(std::exchange(other.pid_, -1)), waited_(other.waited_), ok_(other.ok_) {
        other.waited_ = true;
        other.ok_ = false;
    }
    ChildProcess& operator=(ChildProcess&& other) noexcept {
        if (this != &other) {
            reap();
            pid_ = std::exchange(other.pid_, -1);
            waited_ = other.waited_;
            ok_ = other.ok_;
            other.waited_ = true;
            other.ok_ = false;
        }
        return *this;
    }
    ~ChildProcess() { reap(); }

    // Retries EINTR. Success is a normal exit of 0. A signal is failure.
    bool reap() {
        if (waited_) return ok_;
        waited_ = true;
        if (pid_ <= 0) return ok_ = false;
        const pid_t child = pid_;
        pid_ = -1;
        int status = 0;
        pid_t got = -1;
        do {
            got = ::waitpid(child, &status, 0);
        } while (got < 0 && errno == EINTR);
        ok_ = got == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        return ok_;
    }

private:
    pid_t pid_ = -1;
    bool waited_ = false;
    bool ok_ = false;
};

void gather_rows(int fd, int rank, float* values, int rows) {
    if (fd < 0 || rows <= 0) return;
    const int mid = rows / 2;
    const int mine_begin = rank == 0 ? 0 : mid;
    const int mine_count = rank == 0 ? mid : rows - mid;
    const int other_begin = rank == 0 ? mid : 0;
    const int other_count = rows - mine_count;
    if (rank == 0) {
        send_all(fd, values + mine_begin, static_cast<std::size_t>(mine_count) * sizeof(float));
        recv_all(fd, values + other_begin, static_cast<std::size_t>(other_count) * sizeof(float));
    } else {
        recv_all(fd, values + other_begin, static_cast<std::size_t>(other_count) * sizeof(float));
        send_all(fd, values + mine_begin, static_cast<std::size_t>(mine_count) * sizeof(float));
    }
}

void rmsnorm(const float* x, const float* weight, int n, float eps, float* y) {
    float square = 0.f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const float inv = 1.f / std::sqrt(square / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (weight != nullptr ? weight[i] : 1.f);
}

// TileLang stashes the collapsed stream sum in bf16, forms rsqrt from the fp32
// sum, and stores the layer input in bf16.
void rmsnorm_stash(const float* x, const float* weight, int n, float eps, float* y) {
    float square = 0.f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const float inv = 1.f / std::sqrt(square / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) {
        const float stashed = bf16_to_f32(f32_to_bf16(x[i]));
        const float scaled = stashed * inv * (weight != nullptr ? weight[i] : 1.f);
        y[i] = bf16_to_f32(f32_to_bf16(scaled));
    }
}

struct Tensor {
    const std::byte* data = nullptr;
    std::int64_t shape[8]{};
    int rank = 0;
    char kind = '?';
};

std::int64_t dim_at(const Tensor& tensor, int axis) {
    if (axis < 0 || axis >= tensor.rank) fail("tensor rank is short");
    return tensor.shape[axis];
}

class MappedShard {
public:
    MappedShard() = default;
    MappedShard(void* address, std::size_t length, int fd) noexcept : address_(address), length_(length), fd_(fd) {}
    MappedShard(const MappedShard&) = delete;
    MappedShard& operator=(const MappedShard&) = delete;
    MappedShard(MappedShard&& other) noexcept
        : address_(std::exchange(other.address_, nullptr)),
          length_(std::exchange(other.length_, std::size_t{0})),
          fd_(std::exchange(other.fd_, -1)) {}
    MappedShard& operator=(MappedShard&& other) noexcept {
        if (this != &other) {
            release();
            address_ = std::exchange(other.address_, nullptr);
            length_ = std::exchange(other.length_, std::size_t{0});
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }
    ~MappedShard() { release(); }

    [[nodiscard]] void* address() const { return address_; }
    [[nodiscard]] std::size_t length() const { return length_; }

private:
    void release() noexcept {
        if (address_ != nullptr && address_ != MAP_FAILED && length_ > 0U) ::munmap(address_, length_);
        if (fd_ >= 0) ::close(fd_);
        address_ = nullptr;
        length_ = 0U;
        fd_ = -1;
    }

    void* address_ = nullptr;
    std::size_t length_ = 0;
    int fd_ = -1;
};

class Store {
public:
    explicit Store(const std::filesystem::path& directory) {
        const auto index_path = directory / "model.safetensors.index.json";
        std::unordered_map<std::string, std::string> shard_of;
        if (!std::filesystem::is_regular_file(index_path)) {
            if (!std::filesystem::is_regular_file(directory / "model.safetensors")) {
                fail("missing model.safetensors.index.json");
            }
            shard_of.emplace("model.safetensors", "model.safetensors");
        } else {
        std::ifstream index_file(index_path, std::ios::binary);
        const std::string index{std::istreambuf_iterator<char>(index_file), std::istreambuf_iterator<char>()};
        if (index.empty()) fail("empty safetensors index");
        const auto map_at = index.find("\"weight_map\"");
        if (map_at == std::string::npos) fail("index has no weight_map");
        json_scan::Cursor cursor(index, map_at);
        cursor.parse_string();
        cursor.expect(':');
        cursor.expect('{');
        if (cursor.peek() != '}') {
            while (true) {
                const auto name = cursor.parse_string();
                cursor.expect(':');
                const auto shard = cursor.parse_string();
                shard_of.emplace(name, shard);
                if (cursor.peek() == ',') {
                    cursor.expect(',');
                    continue;
                }
                break;
            }
        }
        cursor.expect('}');
        }

        std::unordered_set<std::string> opened;
        for (const auto& entry : shard_of) {
            if (!opened.insert(entry.second).second) continue;
            const auto path = directory / entry.second;
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) fail("cannot open shard " + path.string());
            struct stat info {};
            if (::fstat(fd, &info) != 0 || info.st_size <= 0) {
                ::close(fd);
                fail("cannot stat shard " + path.string());
            }
            void* address = ::mmap(nullptr, static_cast<std::size_t>(info.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
            if (address == MAP_FAILED) {
                ::close(fd);
                fail("cannot map shard " + path.string());
            }
            MappedShard shard(address, static_cast<std::size_t>(info.st_size), fd);
            maps_.push_back(std::move(shard));
        }

        for (const auto& map : maps_) {
            if (map.length() < 8U) fail("short safetensors file");
            const auto* bytes = static_cast<const unsigned char*>(map.address());
            std::uint64_t header_len = 0;
            for (int i = 0; i < 8; ++i) {
                header_len |= static_cast<std::uint64_t>(bytes[i]) << static_cast<unsigned>(i * 8);
            }
            if (header_len == 0U || 8U + header_len > map.length()) fail("bad safetensors header");
            const std::string_view header(reinterpret_cast<const char*>(bytes + 8), static_cast<std::size_t>(header_len));
            const auto* data = bytes + 8 + header_len;
            json_scan::Cursor header_cursor(header);
            header_cursor.expect('{');
            if (header_cursor.peek() == '}') continue;
            while (true) {
                const auto name = header_cursor.parse_string();
                header_cursor.expect(':');
                if (name == "__metadata__") {
                    header_cursor.skip_value();
                } else {
                    Tensor tensor;
                    header_cursor.expect('{');
                    while (true) {
                        const auto key = header_cursor.parse_string();
                        header_cursor.expect(':');
                        if (key == "dtype") {
                            const auto dtype = header_cursor.parse_string();
                            if (dtype == "BF16") tensor.kind = 'b';
                            else if (dtype == "F32") tensor.kind = 'f';
                            else if (dtype == "U8") tensor.kind = 'u';
                            else if (dtype == "F8_E4M3") tensor.kind = '8';
                            else fail("unsupported dtype " + dtype + " for " + name);
                        } else if (key == "shape") {
                            header_cursor.expect('[');
                            tensor.rank = 0;
                            if (header_cursor.peek() != ']') {
                                while (true) {
                                    if (tensor.rank >= 8) fail("rank above 8 for " + name);
                                    tensor.shape[tensor.rank++] = header_cursor.parse_int();
                                    if (header_cursor.peek() == ',') {
                                        header_cursor.expect(',');
                                        continue;
                                    }
                                    break;
                                }
                            }
                            header_cursor.expect(']');
                        } else if (key == "data_offsets") {
                            header_cursor.expect('[');
                            const auto begin = header_cursor.parse_int();
                            header_cursor.expect(',');
                            header_cursor.parse_int();
                            header_cursor.expect(']');
                            if (begin < 0) fail("negative data offset for " + name);
                            tensor.data = reinterpret_cast<const std::byte*>(data + begin);
                        } else {
                            header_cursor.skip_value();
                        }
                        if (header_cursor.peek() == ',') {
                            header_cursor.expect(',');
                            continue;
                        }
                        header_cursor.expect('}');
                        break;
                    }
                    if (tensor.data == nullptr || tensor.kind == '?') fail("incomplete tensor header for " + name);
                    tensors_.emplace(name, tensor);
                }
                if (header_cursor.peek() == ',') {
                    header_cursor.expect(',');
                    continue;
                }
                header_cursor.expect('}');
                break;
            }
        }
    }

    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;

    [[nodiscard]] const Tensor& get(const std::string& name) const {
        const auto found = tensors_.find(name);
        if (found == tensors_.end()) fail("missing tensor " + name);
        return found->second;
    }

    [[nodiscard]] float scalar(const std::string& name) const {
        const auto& tensor = get(name);
        if (tensor.kind != 'f' || tensor.rank != 0) fail("expected rank-0 F32 for " + name);
        float value = 0.f;
        std::memcpy(&value, tensor.data, sizeof(value));
        return value;
    }

    void load(const std::string& name, float* dest, int n) const {
        const auto& tensor = get(name);
        std::int64_t count = 1;
        for (int axis = 0; axis < tensor.rank; ++axis) count *= tensor.shape[axis];
        if (count != n) fail("vector length mismatch for " + name);
        if (tensor.kind == 'f') {
            std::memcpy(dest, tensor.data, static_cast<std::size_t>(n) * sizeof(float));
            return;
        }
        if (tensor.kind != 'b') fail("cannot load " + name + " as float");
        const auto* bits = reinterpret_cast<const std::uint16_t*>(tensor.data);
        for (int i = 0; i < n; ++i) dest[i] = bf16_to_f32(bits[i]);
    }

private:
    std::vector<MappedShard> maps_;
    std::unordered_map<std::string, Tensor> tensors_;
};

void gemv_bf16(const std::uint16_t* weight, const float* x, int rows, int cols, int row0, int row1, float* y) {
    (void)rows;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = row0; row < row1; ++row) {
        const auto* line = weight + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
        float acc = 0.f;
        for (int col = 0; col < cols; ++col) acc += bf16_to_f32(line[col]) * x[col];
        y[row] = acc;
    }
}

float cast_to_e2m1(float value) {
    const float sign = std::copysign(1.f, value);
    const float magnitude = std::fabs(value);
    float snapped = 6.f;
    if (magnitude <= 0.25f) snapped = 0.f;
    else if (magnitude < 0.75f) snapped = 0.5f;
    else if (magnitude <= 1.25f) snapped = 1.f;
    else if (magnitude < 1.75f) snapped = 1.5f;
    else if (magnitude <= 2.5f) snapped = 2.f;
    else if (magnitude < 3.5f) snapped = 3.f;
    else if (magnitude <= 5.f) snapped = 4.f;
    return sign * snapped;
}

// ModelOpt W4A4. global_scale is 1/input_scale. Each 16-wide group is quantized
// and restored before the weight dot, matching the FP4 GEMM reference.
void quant_dequant_nvfp4_activation(const float* x, int cols, float input_scale, float* y) {
    if (!(input_scale > 0.f) || !std::isfinite(input_scale) || cols % 16 != 0) fail("NVFP4 input scale");
    const float global_scale = 1.f / input_scale;
    for (int group = 0; group < cols / 16; ++group) {
        float rounded[16];
        float peak = 0.f;
        for (int lane = 0; lane < 16; ++lane) {
            rounded[lane] = bf16_to_f32(f32_to_bf16(x[group * 16 + lane]));
            peak = std::max(peak, std::fabs(rounded[lane]));
        }
        float scale = global_scale * (peak / 6.f);
        if (scale > 448.f) scale = 448.f;
        if (scale < -448.f) scale = -448.f;
        scale = fp8_e4m3_to_f32(f32_to_e4m3(scale));
        const float reduced = scale / global_scale;
        const float output_scale = 1.f / (reduced + (reduced == 0.f ? 1.0e8f : 0.f));
        for (int lane = 0; lane < 16; ++lane) {
            float clipped = rounded[lane] * output_scale;
            if (clipped > 6.f) clipped = 6.f;
            if (clipped < -6.f) clipped = -6.f;
            y[group * 16 + lane] = cast_to_e2m1(clipped) * reduced;
        }
    }
}

void gemv_nvfp4_rows(const std::uint8_t* packed, const std::uint8_t* scales, float scale2, const float* x, int rows,
                     int cols, int row0, int row1, float* y) {
    const int packed_cols = cols / 2;
    const int scale_cols = cols / 16;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = row0; row < row1; ++row) {
        const auto* packed_row = packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
        const auto* scale_row = scales + static_cast<std::size_t>(row) * static_cast<std::size_t>(scale_cols);
        float acc = 0.f;
        for (int group = 0; group < scale_cols; ++group) {
            const float scale = fp8_e4m3_to_f32(scale_row[group]) * scale2;
            const auto* bytes = packed_row + group * 8;
            const float* input = x + group * 16;
            for (int byte = 0; byte < 8; ++byte) {
                const std::uint8_t value = bytes[byte];
                acc += e2m1(static_cast<std::uint8_t>(value & 0x0fu)) * scale * input[byte * 2];
                acc += e2m1(static_cast<std::uint8_t>(value >> 4U)) * scale * input[byte * 2 + 1];
            }
        }
        y[row] = acc;
    }
}

class TextModel {
public:
    TextModel(const std::filesystem::path& directory, int world_size)
        : spec_(glm53_flash_spec()), store_(directory), world_(world_size) {
        if (spec_.qk_rope_head_dim != 0U) fail("GLM-5.3-Flash text forward expects RoPE head dim 0");
        if (world_ != 1 && world_ != 2) fail("world size must be 1 or 2");
        hidden_ = static_cast<int>(spec_.hidden_size);
        streams_ = static_cast<int>(spec_.hyper_connection_streams);
        heads_ = static_cast<int>(spec_.linear_attention_heads);
        head_dim_ = static_cast<int>(spec_.linear_attention_head_dim);
        qkv_ = heads_ * head_dim_;
        kernel_ = static_cast<int>(spec_.short_conv_kernel_size);
        attn_heads_ = static_cast<int>(spec_.attention_heads);
        qk_dim_ = static_cast<int>(spec_.qk_head_dim);
        v_dim_ = static_cast<int>(spec_.v_head_dim);
        q_lora_ = static_cast<int>(spec_.q_lora_rank);
        kv_lora_ = static_cast<int>(spec_.kv_lora_rank);
        rms_eps_ = 1e-5f;
        hc_eps_ = static_cast<float>(spec_.hyper_connection_epsilon);
        sinkhorn_ = static_cast<int>(spec_.hyper_connection_sinkhorn_iterations);
        lower_bound_ = static_cast<float>(spec_.kda_gate_lower_bound);
        swiglu_limit_ = static_cast<float>(spec_.swiglu_limit);
        top_k_ = static_cast<int>(spec_.experts_per_token);
        experts_ = static_cast<int>(spec_.routed_experts);
        scaling_ = static_cast<float>(spec_.routed_scaling_factor);
        const int wide = std::max(qkv_, attn_heads_ * (qk_dim_ + v_dim_));
        buf_a_.assign(static_cast<std::size_t>(wide), 0.f);
        buf_b_.assign(static_cast<std::size_t>(wide), 0.f);
        buf_c_.assign(static_cast<std::size_t>(wide), 0.f);
        buf_d_.assign(static_cast<std::size_t>(std::max(wide, static_cast<int>(spec_.intermediate_size))), 0.f);
        streams_state_.assign(static_cast<std::size_t>(streams_ * hidden_), 0.f);
        branch_.assign(static_cast<std::size_t>(hidden_), 0.f);
        collapsed_.assign(static_cast<std::size_t>(hidden_), 0.f);
        final_.assign(static_cast<std::size_t>(std::max(hidden_, qkv_)), 0.f);
        const int layers = static_cast<int>(spec_.layers.size());
        kda_.resize(static_cast<std::size_t>(layers));
        mla_.resize(static_cast<std::size_t>(layers));
        for (int layer = 0; layer < layers; ++layer) {
            if (spec_.layers[static_cast<std::size_t>(layer)].mixer != MixerKind::kKda) continue;
            kda_[static_cast<std::size_t>(layer)].state.assign(static_cast<std::size_t>(heads_ * head_dim_ * head_dim_), 0.f);
            const int hist = qkv_ * (kernel_ - 1);
            kda_[static_cast<std::size_t>(layer)].q_mem.assign(static_cast<std::size_t>(hist), 0.f);
            kda_[static_cast<std::size_t>(layer)].k_mem.assign(static_cast<std::size_t>(hist), 0.f);
            kda_[static_cast<std::size_t>(layer)].v_mem.assign(static_cast<std::size_t>(hist), 0.f);
        }
    }

    void step(std::int32_t token) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= spec_.vocab_size) fail("prompt token is outside the vocabulary");
        const auto& embed = store_.get("model.language_model.embed_tokens.weight");
        if (embed.kind != 'b' || dim_at(embed, 1) != hidden_) fail("embed_tokens layout");
        const auto* row = reinterpret_cast<const std::uint16_t*>(embed.data) +
                          static_cast<std::size_t>(token) * static_cast<std::size_t>(hidden_);
        for (int stream = 0; stream < streams_; ++stream) {
            float* dest = streams_state_.data() + static_cast<std::size_t>(stream * hidden_);
            for (int dim = 0; dim < hidden_; ++dim) dest[dim] = bf16_to_f32(row[dim]);
        }
        for (std::size_t layer = 0; layer < spec_.layers.size(); ++layer) {
            mix(static_cast<int>(layer), spec_.layers[layer].mixer, "hc_attn_", "input_layernorm.weight");
            feed(static_cast<int>(layer), spec_.layers[layer].ffn);
            // Serving kernels store each mHC stream in bf16. fp32 streams
            // move a close argmax once the capture KDA state is present.
            for (float& value : streams_state_) value = bf16_to_f32(f32_to_bf16(value));
            if (capture_taps_) record_tap(static_cast<int>(layer));
        }
        std::vector<float> mean(static_cast<std::size_t>(hidden_), 0.f);
        for (int dim = 0; dim < hidden_; ++dim) {
            float sum = 0.f;
            for (int stream = 0; stream < streams_; ++stream) {
                sum += streams_state_[static_cast<std::size_t>(stream * hidden_ + dim)];
            }
            mean[static_cast<std::size_t>(dim)] = sum / static_cast<float>(streams_);
        }
        std::vector<float> norm(static_cast<std::size_t>(hidden_));
        store_.load("model.language_model.norm.weight", norm.data(), hidden_);
        rmsnorm(mean.data(), norm.data(), hidden_, rms_eps_, final_.data());
        ++seen_;
    }

    [[nodiscard]] std::int32_t argmax() {
        const auto& head = store_.get("lm_head.weight");
        if (head.kind != 'b' || dim_at(head, 0) != static_cast<std::int64_t>(spec_.vocab_size) || dim_at(head, 1) != hidden_) {
            fail("lm_head layout");
        }
        std::vector<float> logits(spec_.vocab_size);
        const int rows = static_cast<int>(spec_.vocab_size);
        apply_rows(0, rows, [&](int row0, int row1) {
            gemv_bf16(reinterpret_cast<const std::uint16_t*>(head.data), final_.data(), rows, hidden_, row0, row1,
                      logits.data());
        });
        if (link_fd_ >= 0) gather_rows(link_fd_, rank_, logits.data(), rows);
        int best = 0;
        int second = 0;
        float best_value = logits[0];
        float second_value = -1e30f;
        for (int row = 1; row < rows; ++row) {
            const float value = logits[static_cast<std::size_t>(row)];
            if (value > best_value) {
                second = best;
                second_value = best_value;
                best_value = value;
                best = row;
            } else if (value > second_value) {
                second_value = value;
                second = row;
            }
        }
        float square = 0.f;
        for (int dim = 0; dim < hidden_; ++dim) square += final_[static_cast<std::size_t>(dim)] * final_[static_cast<std::size_t>(dim)];
        std::cerr << "final_rms=" << std::sqrt(square / static_cast<float>(hidden_)) << " best=" << best
                  << " logit=" << best_value << " second=" << second << " logit2=" << second_value
                  << " id13041=" << logits[13041] << " id198=" << logits[198] << '\n';
        last_margin_ = best_value - second_value;
        if (rank_ == 0) {
            if (const char* dump = std::getenv("NINFER_HIDDEN_DUMP"); dump != nullptr && dump[0] != '\0') {
                std::ofstream out(dump, std::ios::binary | (dumped_ ? std::ios::app : std::ios::trunc));
                out.write(reinterpret_cast<const char*>(final_.data()), static_cast<std::streamsize>(hidden_) * 4);
                dumped_ = true;
            }
        }
        return static_cast<std::int32_t>(best);
    }

    void set_rank(int rank, int fd) {
        rank_ = rank;
        link_fd_ = fd;
        world_ = 2;
    }

    void set_capture(bool enabled) { capture_taps_ = enabled; }

    // CUDA-graph capture fills input ids with token 0. The serving block
    // zeroer clears attention pages and skips mamba, so a fresh request
    // reads that KDA state with an empty MLA cache.
    void prime_capture_kda() {
        step(0);
        for (auto& cache : mla_) {
            cache.latents.clear();
            cache.tokens = 0;
        }
    }

    [[nodiscard]] const float* tap(int index) const { return taps_[static_cast<std::size_t>(index)].data(); }

    [[nodiscard]] float margin() const { return last_margin_; }

    struct KdaCache {
        std::vector<float> state;
        std::vector<float> q_mem;
        std::vector<float> k_mem;
        std::vector<float> v_mem;
    };
    struct MlaCache {
        std::vector<float> latents;
        int tokens = 0;
    };
    struct Snapshot {
        std::vector<KdaCache> kda;
        std::vector<MlaCache> mla;
        int seen = 0;
    };

    [[nodiscard]] Snapshot save() const {
        Snapshot snap;
        snap.kda = kda_;
        snap.mla = mla_;
        snap.seen = seen_;
        return snap;
    }

    void restore(const Snapshot& snap) {
        kda_ = snap.kda;
        mla_ = snap.mla;
        seen_ = snap.seen;
    }

    void embed_row(std::int32_t token, float* dest) const {
        const auto& embed = store_.get("model.language_model.embed_tokens.weight");
        const auto* row = reinterpret_cast<const std::uint16_t*>(embed.data) +
                          static_cast<std::size_t>(token) * static_cast<std::size_t>(hidden_);
        for (int dim = 0; dim < hidden_; ++dim) dest[dim] = bf16_to_f32(row[dim]);
    }

    void lm_head_logits(const float* hidden, float* logits) {
        const auto& head = store_.get("lm_head.weight");
        const int rows = static_cast<int>(spec_.vocab_size);
        apply_rows(0, rows, [&](int row0, int row1) {
            gemv_bf16(reinterpret_cast<const std::uint16_t*>(head.data), hidden, rows, hidden_, row0, row1, logits);
        });
        if (link_fd_ >= 0) gather_rows(link_fd_, rank_, logits, rows);
    }

private:
    void dump_prompt_layer(int layer, const char* tag, const float* data, int count) {
        const bool conv_tag = std::strncmp(tag, "conv-", 5) == 0;
        const bool prompt = seen_ == 1 && layer <= 3;
        const bool prime_conv = seen_ == 0 && conv_tag && layer == 0;
        const bool generated = seen_ == 2 && layer == 0;
        if (!prompt && !prime_conv && !generated) return;
        const char* dir = std::getenv("NINFER_LAYER_DUMP");
        if (dir == nullptr || dir[0] == '\0') return;
        double square = 0.0;
        for (int i = 0; i < count; ++i) square += static_cast<double>(data[i]) * static_cast<double>(data[i]);
        std::cerr << "tap L" << layer << " s" << seen_ << ' ' << tag << " nrm=" << std::sqrt(square) << " y0=" << data[0] << '\n';
        const std::string path = std::string(dir) + "/L" + std::to_string(layer) + "-s" + std::to_string(seen_) + "-" + tag + ".bin";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count) * 4);
    }

    void record_tap(int layer) {
        const int ids[5] = {5, 14, 24, 33, 42};
        for (int index = 0; index < 5; ++index) {
            if (layer != ids[index]) continue;
            auto& dest = taps_[static_cast<std::size_t>(index)];
            const std::size_t at = dest.size();
            dest.resize(at + static_cast<std::size_t>(hidden_));
            for (int dim = 0; dim < hidden_; ++dim) {
                float sum = 0.f;
                for (int stream = 0; stream < streams_; ++stream) {
                    sum += streams_state_[static_cast<std::size_t>(stream * hidden_ + dim)];
                }
                dest[at + static_cast<std::size_t>(dim)] = sum / static_cast<float>(streams_);
            }
        }
    }

    template <typename Fn>
    void apply_rows(int rows_unused, int rows, Fn&& fn) {
        (void)rows_unused;
        if (link_fd_ >= 0) {
            const int mid = rows / 2;
            if (rank_ == 0) fn(0, mid);
            else fn(mid, rows);
            return;
        }
        if (world_ == 1) {
            fn(0, rows);
            return;
        }
        const int mid = rows / 2;
        fn(0, mid);
        fn(mid, rows);
    }

    void linear_bf16(const std::string& name, const float* x, int cols, float* y, int rows) {
        const auto& tensor = store_.get(name);
        if (tensor.kind != 'b' || tensor.rank != 2 || dim_at(tensor, 0) != rows || dim_at(tensor, 1) != cols) {
            fail("BF16 linear layout " + name);
        }
        apply_rows(0, rows, [&](int row0, int row1) {
            gemv_bf16(reinterpret_cast<const std::uint16_t*>(tensor.data), x, rows, cols, row0, row1, y);
        });
        if (link_fd_ >= 0) gather_rows(link_fd_, rank_, y, rows);
    }

    void linear_nvfp4(const std::string& prefix, const float* x, int cols, float* y, int rows) {
        const auto& packed = store_.get(prefix + ".weight");
        const auto& scales = store_.get(prefix + ".weight_scale");
        const float scale2 = store_.scalar(prefix + ".weight_scale_2");
        const float input_scale = store_.scalar(prefix + ".input_scale");
        if (packed.kind != 'u' || scales.kind != '8' || packed.rank != 2 || scales.rank != 2) {
            fail("NVFP4 layout " + prefix);
        }
        if (dim_at(packed, 0) != rows || dim_at(packed, 1) != cols / 2 || dim_at(scales, 0) != rows ||
            dim_at(scales, 1) != cols / 16) {
            fail("NVFP4 shape " + prefix);
        }
        // Dense MLP and routed experts both use ModelOpt W4A4. The block
        // scales stay in checkpoint order; weight_scale_2 multiplies them.
        std::vector<float> quantized(static_cast<std::size_t>(cols));
        quant_dequant_nvfp4_activation(x, cols, input_scale, quantized.data());
        apply_rows(0, rows, [&](int row0, int row1) {
            gemv_nvfp4_rows(reinterpret_cast<const std::uint8_t*>(packed.data),
                            reinterpret_cast<const std::uint8_t*>(scales.data), scale2, quantized.data(), rows, cols,
                            row0, row1, y);
        });
        if (link_fd_ >= 0) gather_rows(link_fd_, rank_, y, rows);
        for (int row = 0; row < rows; ++row) y[row] = bf16_to_f32(f32_to_bf16(y[row]));
    }

    void project_hc(const std::string& name_prefix, float* post, float* comb) {
        const int coeff = (streams_ + 2) * streams_;
        const int flat = streams_ * hidden_;
        std::vector<float> fn(static_cast<std::size_t>(coeff * flat));
        std::vector<float> base(static_cast<std::size_t>(coeff));
        std::vector<float> scale(3);
        store_.load(name_prefix + "fn", fn.data(), coeff * flat);
        store_.load(name_prefix + "base", base.data(), coeff);
        store_.load(name_prefix + "scale", scale.data(), 3);
        mhc_project(streams_state_.data(), streams_, hidden_, fn.data(), base.data(), scale.data(), rms_eps_, hc_eps_,
                    sinkhorn_, post, comb, collapsed_.data());
    }

    void mix(int layer, MixerKind mixer, const char* hc_name, const char* norm_name) {
        const std::string prefix = layer_prefix(layer);
        std::vector<float> post(static_cast<std::size_t>(streams_));
        std::vector<float> comb(static_cast<std::size_t>(streams_ * streams_));
        std::vector<float> residual = streams_state_;
        project_hc(prefix + hc_name, post.data(), comb.data());
        std::vector<float> norm(static_cast<std::size_t>(hidden_));
        store_.load(prefix + norm_name, norm.data(), hidden_);
        rmsnorm_stash(collapsed_.data(), norm.data(), hidden_, rms_eps_, branch_.data());
        dump_prompt_layer(layer, "attn-in", branch_.data(), hidden_);
        if (mixer == MixerKind::kKda) kda(layer, branch_.data(), buf_a_.data());
        else mla(layer, branch_.data(), buf_a_.data());
        // o_proj is stored bf16 before hc_post. The fp32 tail is not consumed.
        for (int dim = 0; dim < hidden_; ++dim) {
            buf_a_[static_cast<std::size_t>(dim)] = bf16_to_f32(f32_to_bf16(buf_a_[static_cast<std::size_t>(dim)]));
        }
        dump_prompt_layer(layer, "attn", buf_a_.data(), hidden_);
        mhc_combine(residual.data(), buf_a_.data(), post.data(), comb.data(), streams_, hidden_, streams_state_.data());
        // The fused hc_pre reads the bf16 hc_post residual.
        for (float& value : streams_state_) value = bf16_to_f32(f32_to_bf16(value));
    }

    void feed(int layer, FfnKind ffn) {
        const std::string prefix = layer_prefix(layer);
        std::vector<float> post(static_cast<std::size_t>(streams_));
        std::vector<float> comb(static_cast<std::size_t>(streams_ * streams_));
        std::vector<float> residual = streams_state_;
        project_hc(prefix + "hc_ffn_", post.data(), comb.data());
        std::vector<float> norm(static_cast<std::size_t>(hidden_));
        store_.load(prefix + "post_attention_layernorm.weight", norm.data(), hidden_);
        rmsnorm_stash(collapsed_.data(), norm.data(), hidden_, rms_eps_, branch_.data());
        dump_prompt_layer(layer, "mlp-in", branch_.data(), hidden_);
        if (ffn == FfnKind::kDense) dense_mlp(prefix, branch_.data(), buf_a_.data());
        else moe(prefix, branch_.data(), buf_a_.data());
        dump_prompt_layer(layer, "ffn", buf_a_.data(), hidden_);
        mhc_combine(residual.data(), buf_a_.data(), post.data(), comb.data(), streams_, hidden_, streams_state_.data());
    }

    void dense_mlp(const std::string& prefix, const float* x, float* y) {
        const int intermediate = static_cast<int>(spec_.intermediate_size);
        linear_nvfp4(prefix + "mlp.gate_proj", x, hidden_, buf_b_.data(), intermediate);
        linear_nvfp4(prefix + "mlp.up_proj", x, hidden_, buf_c_.data(), intermediate);
        swiglu_clamp(buf_b_.data(), buf_c_.data(), intermediate, swiglu_limit_, buf_d_.data());
        linear_nvfp4(prefix + "mlp.down_proj", buf_d_.data(), intermediate, y, hidden_);
    }

    void mlp_bf16(const std::string& prefix, const float* x, int intermediate, float* y) {
        linear_bf16(prefix + "gate_proj.weight", x, hidden_, buf_b_.data(), intermediate);
        linear_bf16(prefix + "up_proj.weight", x, hidden_, buf_c_.data(), intermediate);
        swiglu_clamp(buf_b_.data(), buf_c_.data(), intermediate, swiglu_limit_, buf_d_.data());
        linear_bf16(prefix + "down_proj.weight", buf_d_.data(), intermediate, y, hidden_);
    }

    void moe(const std::string& prefix, const float* x, float* y) {
        const int intermediate = static_cast<int>(spec_.moe_intermediate_size);
        linear_bf16(prefix + "mlp.gate.weight", x, hidden_, buf_b_.data(), experts_);
        std::vector<float> bias(static_cast<std::size_t>(experts_));
        store_.load(prefix + "mlp.gate.e_score_correction_bias", bias.data(), experts_);
        std::vector<int> indices(static_cast<std::size_t>(top_k_));
        std::vector<float> weights(static_cast<std::size_t>(top_k_));
        router_select(buf_b_.data(), bias.data(), experts_, top_k_, scaling_, true, indices.data(), weights.data());
        if (seen_ == 1) {
            if (const char* dir = std::getenv("NINFER_LAYER_DUMP"); dir != nullptr && dir[0] != '\0') {
                const auto mark = prefix.find("layers.");
                const int layer_id = mark == std::string::npos ? -1 : std::atoi(prefix.c_str() + mark + 7);
                std::cerr << "route L" << layer_id;
                for (int pick = 0; pick < top_k_; ++pick) {
                    std::cerr << ' ' << indices[static_cast<std::size_t>(pick)] << ':'
                              << weights[static_cast<std::size_t>(pick)];
                }
                if (layer_id == 4 || layer_id == 9) {
                    const int watch = layer_id == 4 ? 86 : 11;
                    const int eighth = indices[static_cast<std::size_t>(top_k_ - 1)];
                    const float left = sigmoid(buf_b_[watch]) + bias[static_cast<std::size_t>(watch)];
                    const float right = sigmoid(buf_b_[static_cast<std::size_t>(eighth)]) + bias[static_cast<std::size_t>(eighth)];
                    std::cerr << " watch=" << watch << " c=" << left << " eighth=" << eighth << " c8=" << right;
                }
                std::cerr << '\n';
            }
        }
        std::fill(y, y + hidden_, 0.f);
        for (int pick = 0; pick < top_k_; ++pick) {
            const std::string expert = prefix + "mlp.experts." + std::to_string(indices[static_cast<std::size_t>(pick)]) + ".";
            linear_nvfp4(expert + "gate_proj", x, hidden_, buf_b_.data(), intermediate);
            linear_nvfp4(expert + "up_proj", x, hidden_, buf_c_.data(), intermediate);
            swiglu_clamp(buf_b_.data(), buf_c_.data(), intermediate, swiglu_limit_, buf_d_.data());
            linear_nvfp4(expert + "down_proj", buf_d_.data(), intermediate, buf_c_.data(), hidden_);
            const float weight = weights[static_cast<std::size_t>(pick)];
            for (int dim = 0; dim < hidden_; ++dim) y[dim] += weight * buf_c_[static_cast<std::size_t>(dim)];
        }
        mlp_bf16(prefix + "mlp.shared_experts.", x, intermediate, buf_c_.data());
        for (int dim = 0; dim < hidden_; ++dim) y[dim] += buf_c_[static_cast<std::size_t>(dim)];
    }

    void kda(int layer, const float* x, float* y) {
        const std::string prefix = layer_prefix(layer) + "self_attn.";
        auto& cache = kda_[static_cast<std::size_t>(layer)];
        linear_bf16(prefix + "q_proj.weight", x, hidden_, buf_b_.data(), qkv_);
        linear_bf16(prefix + "k_proj.weight", x, hidden_, buf_c_.data(), qkv_);
        linear_bf16(prefix + "v_proj.weight", x, hidden_, buf_d_.data(), qkv_);
        for (int lane = 0; lane < qkv_; ++lane) {
            buf_b_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_b_[static_cast<std::size_t>(lane)]));
            buf_c_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_c_[static_cast<std::size_t>(lane)]));
            buf_d_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_d_[static_cast<std::size_t>(lane)]));
        }
        const auto& q_weight = store_.get(prefix + "q_conv1d.weight");
        const auto& k_weight = store_.get(prefix + "k_conv1d.weight");
        const auto& v_weight = store_.get(prefix + "v_conv1d.weight");
        if (q_weight.kind != 'f' || dim_at(q_weight, 0) != qkv_ || dim_at(q_weight, 2) != kernel_) {
            fail("q conv layout");
        }
        causal_conv_silu(buf_b_.data(), reinterpret_cast<const float*>(q_weight.data), cache.q_mem.data(), qkv_, kernel_,
                         buf_b_.data());
        causal_conv_silu(buf_c_.data(), reinterpret_cast<const float*>(k_weight.data), cache.k_mem.data(), qkv_, kernel_,
                         buf_c_.data());
        causal_conv_silu(buf_d_.data(), reinterpret_cast<const float*>(v_weight.data), cache.v_mem.data(), qkv_, kernel_,
                         buf_d_.data());
        // The short-conv kernel stores silu(conv) as bf16. The fp32 tail is not consumed.
        for (int lane = 0; lane < qkv_; ++lane) {
            buf_b_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_b_[static_cast<std::size_t>(lane)]));
            buf_c_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_c_[static_cast<std::size_t>(lane)]));
            buf_d_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_d_[static_cast<std::size_t>(lane)]));
        }
        dump_prompt_layer(layer, "conv-q", buf_b_.data(), qkv_);
        dump_prompt_layer(layer, "conv-k", buf_c_.data(), qkv_);
        dump_prompt_layer(layer, "conv-v", buf_d_.data(), qkv_);
        linear_bf16(prefix + "f_a_proj.weight", x, hidden_, buf_a_.data(), head_dim_);
        for (int lane = 0; lane < head_dim_; ++lane) {
            buf_a_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_a_[static_cast<std::size_t>(lane)]));
        }
        linear_bf16(prefix + "f_b_proj.weight", buf_a_.data(), head_dim_, final_.data(), qkv_);
        std::vector<float> dt_bias(static_cast<std::size_t>(qkv_));
        std::vector<float> a_log(static_cast<std::size_t>(heads_));
        std::vector<float> forget(static_cast<std::size_t>(qkv_));
        store_.load(prefix + "dt_bias", dt_bias.data(), qkv_);
        store_.load(prefix + "A_log", a_log.data(), heads_);
        for (int lane = 0; lane < qkv_; ++lane) {
            final_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(final_[static_cast<std::size_t>(lane)]));
        }
        kda_forget_gate(final_.data(), dt_bias.data(), a_log.data(), lower_bound_, heads_, head_dim_, forget.data());
        linear_bf16(prefix + "b_proj.weight", x, hidden_, buf_a_.data(), heads_);
        for (int head = 0; head < heads_; ++head) {
            buf_a_[static_cast<std::size_t>(head)] =
                sigmoid(bf16_to_f32(f32_to_bf16(buf_a_[static_cast<std::size_t>(head)])));
        }
        kda_recurrent_heads(cache.state.data(), buf_b_.data(), buf_c_.data(), buf_d_.data(), forget.data(), buf_a_.data(),
                            heads_, head_dim_, buf_b_.data());
        // The kernel stores the recurrent output as bf16. The fp32 tail is the
        // whole gap versus the official layer-0 dump once q/k/v match.
        for (int lane = 0; lane < qkv_; ++lane) {
            buf_b_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_b_[static_cast<std::size_t>(lane)]));
        }
        dump_prompt_layer(layer, "recur", buf_b_.data(), qkv_);
        linear_bf16(prefix + "g_a_proj.weight", x, hidden_, buf_a_.data(), head_dim_);
        for (int lane = 0; lane < head_dim_; ++lane) {
            buf_a_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_a_[static_cast<std::size_t>(lane)]));
        }
        linear_bf16(prefix + "g_b_proj.weight", buf_a_.data(), head_dim_, buf_c_.data(), qkv_);
        for (int lane = 0; lane < qkv_; ++lane) {
            buf_c_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_c_[static_cast<std::size_t>(lane)]));
        }
        std::vector<float> o_norm(static_cast<std::size_t>(head_dim_));
        store_.load(prefix + "o_norm.weight", o_norm.data(), head_dim_);
        for (int head = 0; head < heads_; ++head) {
            float* row = buf_b_.data() + static_cast<std::size_t>(head * head_dim_);
            const float* gate = buf_c_.data() + static_cast<std::size_t>(head * head_dim_);
            rmsnorm(row, o_norm.data(), head_dim_, rms_eps_, row);
            for (int dim = 0; dim < head_dim_; ++dim) row[dim] *= sigmoid(gate[dim]);
        }
        for (int lane = 0; lane < qkv_; ++lane) {
            buf_b_[static_cast<std::size_t>(lane)] = bf16_to_f32(f32_to_bf16(buf_b_[static_cast<std::size_t>(lane)]));
        }
        dump_prompt_layer(layer, "oproj-in", buf_b_.data(), qkv_);
        linear_bf16(prefix + "o_proj.weight", buf_b_.data(), qkv_, y, hidden_);
    }

    void mla(int layer, const float* x, float* y) {
        const std::string prefix = layer_prefix(layer) + "self_attn.";
        linear_bf16(prefix + "q_a_proj.weight", x, hidden_, buf_b_.data(), q_lora_);
        std::vector<float> q_norm(static_cast<std::size_t>(q_lora_));
        store_.load(prefix + "q_a_layernorm.weight", q_norm.data(), q_lora_);
        rmsnorm(buf_b_.data(), q_norm.data(), q_lora_, rms_eps_, buf_b_.data());
        linear_bf16(prefix + "q_b_proj.weight", buf_b_.data(), q_lora_, buf_c_.data(), attn_heads_ * qk_dim_);
        linear_bf16(prefix + "kv_a_proj_with_mqa.weight", x, hidden_, buf_a_.data(), kv_lora_);
        std::vector<float> kv_norm(static_cast<std::size_t>(kv_lora_));
        store_.load(prefix + "kv_a_layernorm.weight", kv_norm.data(), kv_lora_);
        rmsnorm(buf_a_.data(), kv_norm.data(), kv_lora_, rms_eps_, buf_a_.data());
        // fp8_ds_mla stores each 128-wide latent group as e4m3 with scale amax/448.
        if (kv_lora_ % 128 == 0) {
            for (int block = 0; block < kv_lora_; block += 128) {
                float peak = 0.f;
                for (int lane = 0; lane < 128; ++lane) {
                    peak = std::max(peak, std::fabs(buf_a_[static_cast<std::size_t>(block + lane)]));
                }
                if (peak == 0.f) continue;
                const float scale = peak / 448.f;
                for (int lane = 0; lane < 128; ++lane) {
                    float& slot = buf_a_[static_cast<std::size_t>(block + lane)];
                    slot = fp8_e4m3_to_f32(f32_to_e4m3(slot / scale)) * scale;
                }
            }
        }
        auto& cache = mla_[static_cast<std::size_t>(layer)];
        cache.latents.insert(cache.latents.end(), buf_a_.begin(), buf_a_.begin() + kv_lora_);
        ++cache.tokens;
        // Prompts no longer than index_topk select every causal token, so the mask is dense causal.
        if (cache.tokens > static_cast<int>(spec_.index_topk)) fail("sequence exceeds the dense-causal indexer bound");
        const float scale = 1.f / std::sqrt(static_cast<float>(qk_dim_));
        std::vector<float> scores(static_cast<std::size_t>(attn_heads_ * cache.tokens), 0.f);
        std::vector<float> values(static_cast<std::size_t>(cache.tokens * attn_heads_ * v_dim_), 0.f);
        for (int token = 0; token < cache.tokens; ++token) {
            const float* latent = cache.latents.data() + static_cast<std::size_t>(token * kv_lora_);
            linear_bf16(prefix + "kv_b_proj.weight", latent, kv_lora_, buf_d_.data(), attn_heads_ * (qk_dim_ + v_dim_));
            for (int head = 0; head < attn_heads_; ++head) {
                const float* query = buf_c_.data() + static_cast<std::size_t>(head * qk_dim_);
                const float* key = buf_d_.data() + static_cast<std::size_t>(head * (qk_dim_ + v_dim_));
                const float* value = key + qk_dim_;
                float dot = 0.f;
                for (int dim = 0; dim < qk_dim_; ++dim) dot += query[dim] * key[dim];
                scores[static_cast<std::size_t>(head * cache.tokens + token)] = dot * scale;
                float* dest = values.data() + static_cast<std::size_t>((token * attn_heads_ + head) * v_dim_);
                std::memcpy(dest, value, static_cast<std::size_t>(v_dim_) * sizeof(float));
            }
        }
        for (int head = 0; head < attn_heads_; ++head) {
            float max_score = scores[static_cast<std::size_t>(head * cache.tokens)];
            for (int token = 1; token < cache.tokens; ++token) {
                max_score = std::max(max_score, scores[static_cast<std::size_t>(head * cache.tokens + token)]);
            }
            float sum = 0.f;
            for (int token = 0; token < cache.tokens; ++token) {
                float& score = scores[static_cast<std::size_t>(head * cache.tokens + token)];
                score = std::exp(score - max_score);
                sum += score;
            }
            float* out = buf_b_.data() + static_cast<std::size_t>(head * v_dim_);
            for (int dim = 0; dim < v_dim_; ++dim) out[dim] = 0.f;
            for (int token = 0; token < cache.tokens; ++token) {
                const float prob = scores[static_cast<std::size_t>(head * cache.tokens + token)] / sum;
                const float* value = values.data() + static_cast<std::size_t>((token * attn_heads_ + head) * v_dim_);
                for (int dim = 0; dim < v_dim_; ++dim) out[dim] += prob * value[dim];
            }
        }
        linear_bf16(prefix + "o_proj.weight", buf_b_.data(), attn_heads_ * v_dim_, y, hidden_);
    }

    static std::string layer_prefix(int layer) {
        return "model.language_model.layers." + std::to_string(layer) + ".";
    }

    const ModelSpec& spec_;
    Store store_;
    int world_ = 1;
    int hidden_ = 0;
    int streams_ = 0;
    int heads_ = 0;
    int head_dim_ = 0;
    int qkv_ = 0;
    int kernel_ = 0;
    int attn_heads_ = 0;
    int qk_dim_ = 0;
    int v_dim_ = 0;
    int q_lora_ = 0;
    int kv_lora_ = 0;
    int sinkhorn_ = 0;
    int top_k_ = 0;
    int experts_ = 0;
    int seen_ = 0;
    int rank_ = 0;
    int link_fd_ = -1;
    bool capture_taps_ = false;
    bool dumped_ = false;
    float last_margin_ = 0.f;
    float rms_eps_ = 0.f;
    float hc_eps_ = 0.f;
    float lower_bound_ = 0.f;
    float swiglu_limit_ = 0.f;
    float scaling_ = 0.f;
    std::vector<float> buf_a_;
    std::vector<float> buf_b_;
    std::vector<float> buf_c_;
    std::vector<float> buf_d_;
    std::vector<float> streams_state_;
    std::vector<float> branch_;
    std::vector<float> collapsed_;
    std::vector<float> final_;
    std::vector<KdaCache> kda_;
    std::vector<MlaCache> mla_;
    std::vector<float> taps_[5];
};

constexpr int kDraftHidden = 4096;
constexpr int kDraftHeads = 32;
constexpr int kDraftKv = 8;
constexpr int kDraftDim = 128;
constexpr int kDraftInter = 12288;
constexpr int kMaskToken = 154856;

void rope_heads(float* heads, int count, int dim, int position) {
    constexpr float theta = 10000.f;
    for (int head = 0; head < count; ++head) {
        float* values = heads + static_cast<std::size_t>(head * dim);
        for (int i = 0; i < dim / 2; ++i) {
            const float freq = std::pow(theta, -2.f * static_cast<float>(i) / static_cast<float>(dim));
            const float angle = static_cast<float>(position) * freq;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float first = values[i];
            const float second = values[dim / 2 + i];
            values[i] = first * cosine - second * sine;
            values[dim / 2 + i] = first * sine + second * cosine;
        }
    }
}

void grouped_conv(const float* hidden, const float* dynamic, const float* base, int length, float* out) {
    constexpr int group = 16;
    constexpr int groups = kDraftHidden / group;
    constexpr int kernel = 2;
    std::fill(out, out + static_cast<std::size_t>(length * kDraftHidden), 0.f);
    for (int offset = 0; offset < kernel; ++offset) {
        for (int token = 0; token < length; ++token) {
            const int source = token - offset;
            for (int g = 0; g < groups; ++g) {
                const float scale = dynamic[static_cast<std::size_t>((token * kernel + offset) * groups + g)];
                for (int lane = 0; lane < group; ++lane) {
                    const int column = g * group + lane;
                    const float value = source < 0 ? 0.f : hidden[static_cast<std::size_t>(source * kDraftHidden + column)];
                    const float weight = base[static_cast<std::size_t>(offset * kDraftHidden + column)];
                    out[static_cast<std::size_t>(token * kDraftHidden + column)] += (weight + scale) * value;
                }
            }
        }
    }
}

void project_rows(const Store& store, const std::string& name, const float* input, int length, int cols, int rows,
                  float* output) {
    const auto& tensor = store.get(name);
    for (int token = 0; token < length; ++token) {
        gemv_bf16(reinterpret_cast<const std::uint16_t*>(tensor.data), input + static_cast<std::size_t>(token * cols), rows,
                  cols, 0, rows, output + static_cast<std::size_t>(token * rows));
    }
}

void draft_attention(const Store& store, const std::string& prefix, const float* context, int context_length,
                     const float* queries, int query_length, float* output) {
    const int kv_length = context_length + query_length;
    std::vector<float> query(static_cast<std::size_t>(query_length * kDraftHeads * kDraftDim));
    std::vector<float> key(static_cast<std::size_t>(kv_length * kDraftKv * kDraftDim));
    std::vector<float> value(key.size());
    project_rows(store, prefix + "self_attn.q_proj.weight", queries, query_length, kDraftHidden, kDraftHeads * kDraftDim,
                 query.data());
    std::vector<float> key_context(static_cast<std::size_t>(context_length * kDraftKv * kDraftDim));
    std::vector<float> key_noise(static_cast<std::size_t>(query_length * kDraftKv * kDraftDim));
    std::vector<float> value_context(key_context.size());
    std::vector<float> value_noise(key_noise.size());
    project_rows(store, prefix + "self_attn.k_proj.weight", context, context_length, kDraftHidden, kDraftKv * kDraftDim,
                 key_context.data());
    project_rows(store, prefix + "self_attn.k_proj.weight", queries, query_length, kDraftHidden, kDraftKv * kDraftDim,
                 key_noise.data());
    project_rows(store, prefix + "self_attn.v_proj.weight", context, context_length, kDraftHidden, kDraftKv * kDraftDim,
                 value_context.data());
    project_rows(store, prefix + "self_attn.v_proj.weight", queries, query_length, kDraftHidden, kDraftKv * kDraftDim,
                 value_noise.data());
    std::copy(key_context.begin(), key_context.end(), key.begin());
    std::copy(key_noise.begin(), key_noise.end(), key.begin() + key_context.size());
    std::copy(value_context.begin(), value_context.end(), value.begin());
    std::copy(value_noise.begin(), value_noise.end(), value.begin() + value_context.size());
    std::vector<float> q_norm(kDraftDim);
    std::vector<float> k_norm(kDraftDim);
    store.load(prefix + "self_attn.q_norm.weight", q_norm.data(), kDraftDim);
    store.load(prefix + "self_attn.k_norm.weight", k_norm.data(), kDraftDim);
    for (int token = 0; token < query_length; ++token) {
        for (int head = 0; head < kDraftHeads; ++head) {
            float* row = query.data() + static_cast<std::size_t>((token * kDraftHeads + head) * kDraftDim);
            rmsnorm(row, q_norm.data(), kDraftDim, 1e-5f, row);
        }
        rope_heads(query.data() + static_cast<std::size_t>(token * kDraftHeads * kDraftDim), kDraftHeads, kDraftDim,
                   context_length + token);
    }
    for (int token = 0; token < kv_length; ++token) {
        for (int head = 0; head < kDraftKv; ++head) {
            float* row = key.data() + static_cast<std::size_t>((token * kDraftKv + head) * kDraftDim);
            rmsnorm(row, k_norm.data(), kDraftDim, 1e-5f, row);
        }
        rope_heads(key.data() + static_cast<std::size_t>(token * kDraftKv * kDraftDim), kDraftKv, kDraftDim, token);
    }
    const float scale = 1.f / std::sqrt(static_cast<float>(kDraftDim));
    std::vector<float> mixed(static_cast<std::size_t>(query_length * kDraftHeads * kDraftDim), 0.f);
    for (int token = 0; token < query_length; ++token) {
        for (int head = 0; head < kDraftHeads; ++head) {
            const int kv_head = head / (kDraftHeads / kDraftKv);
            const float* q = query.data() + static_cast<std::size_t>((token * kDraftHeads + head) * kDraftDim);
            std::vector<float> weights(static_cast<std::size_t>(kv_length));
            float max_score = -1e30f;
            for (int key_index = 0; key_index < kv_length; ++key_index) {
                const float* k = key.data() + static_cast<std::size_t>((key_index * kDraftKv + kv_head) * kDraftDim);
                float dot = 0.f;
                for (int dim = 0; dim < kDraftDim; ++dim) dot += q[dim] * k[dim];
                weights[static_cast<std::size_t>(key_index)] = dot * scale;
                max_score = std::max(max_score, weights[static_cast<std::size_t>(key_index)]);
            }
            float sum = 0.f;
            for (float& weight : weights) {
                weight = std::exp(weight - max_score);
                sum += weight;
            }
            float* dest = mixed.data() + static_cast<std::size_t>((token * kDraftHeads + head) * kDraftDim);
            for (int key_index = 0; key_index < kv_length; ++key_index) {
                const float prob = weights[static_cast<std::size_t>(key_index)] / sum;
                const float* v = value.data() + static_cast<std::size_t>((key_index * kDraftKv + kv_head) * kDraftDim);
                for (int dim = 0; dim < kDraftDim; ++dim) dest[dim] += prob * v[dim];
            }
        }
    }
    project_rows(store, prefix + "self_attn.o_proj.weight", mixed.data(), query_length, kDraftHeads * kDraftDim,
                 kDraftHidden, output);
}

void dynamic_conv(const Store& store, const std::string& prefix, const float* input, int length, bool prepare,
                  float* output, std::vector<float>* saved_dynamic) {
    constexpr int groups = kDraftHidden / 16;
    constexpr int kernel = 2;
    constexpr int projected = 2 * kernel * groups;
    std::vector<float> proj(static_cast<std::size_t>(length * projected));
    project_rows(store, prefix + "kernel_projection.weight", input, length, kDraftHidden, projected, proj.data());
    std::vector<float> dynamic(static_cast<std::size_t>(length * kernel * groups));
    std::vector<float> later(dynamic.size());
    for (int token = 0; token < length; ++token) {
        for (int offset = 0; offset < kernel; ++offset) {
            for (int group = 0; group < groups; ++group) {
                const int now = ((0 * kernel + offset) * groups) + group;
                const int after = ((1 * kernel + offset) * groups) + group;
                dynamic[static_cast<std::size_t>((token * kernel + offset) * groups + group)] =
                    proj[static_cast<std::size_t>(token * projected + now)];
                later[static_cast<std::size_t>((token * kernel + offset) * groups + group)] =
                    proj[static_cast<std::size_t>(token * projected + after)];
            }
        }
    }
    if (saved_dynamic != nullptr) *saved_dynamic = std::move(later);
    std::vector<float> base(static_cast<std::size_t>(2 * kernel * kDraftHidden));
    store.load(prefix + "base_kernel", base.data(), static_cast<int>(base.size()));
    const float* base_row = base.data() + static_cast<std::size_t>((prepare ? 0 : 1) * kernel * kDraftHidden);
    grouped_conv(input, dynamic.data(), base_row, length, output);
    (void)prepare;
}

std::vector<std::int32_t> propose_dflash(TextModel& target, const Store& draft, std::int32_t anchor) {
    const int context = static_cast<int>(target.tap(0) == nullptr ? 0 : 1);
    if (context != 1) fail("DFlash context is the one-token prefill");
    std::vector<float> fused(static_cast<std::size_t>(5 * kDraftHidden));
    for (int index = 0; index < 5; ++index) {
        std::copy(target.tap(index), target.tap(index) + kDraftHidden,
                  fused.begin() + static_cast<std::ptrdiff_t>(index * kDraftHidden));
    }
    std::vector<float> context_hidden(kDraftHidden);
    const auto& fc = draft.get("fc.weight");
    gemv_bf16(reinterpret_cast<const std::uint16_t*>(fc.data), fused.data(), kDraftHidden, 5 * kDraftHidden, 0,
              kDraftHidden, context_hidden.data());
    std::vector<float> hidden_norm(kDraftHidden);
    draft.load("hidden_norm.weight", hidden_norm.data(), kDraftHidden);
    rmsnorm(context_hidden.data(), hidden_norm.data(), kDraftHidden, 1e-5f, context_hidden.data());

    constexpr int block = 8;
    std::vector<float> hidden(static_cast<std::size_t>(block * kDraftHidden));
    target.embed_row(anchor, hidden.data());
    for (int token = 1; token < block; ++token) target.embed_row(kMaskToken, hidden.data() + token * kDraftHidden);
    for (int layer = 0; layer < 5; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        std::vector<float> norm(kDraftHidden);
        std::vector<float> normed(hidden.size());
        draft.load(prefix + "input_layernorm.weight", norm.data(), kDraftHidden);
        for (int token = 0; token < block; ++token) {
            rmsnorm(hidden.data() + token * kDraftHidden, norm.data(), kDraftHidden, 1e-5f,
                    normed.data() + token * kDraftHidden);
        }
        std::vector<float> prepared(hidden.size());
        std::vector<float> saved;
        dynamic_conv(draft, prefix + "attention_conv.", normed.data(), block, true, prepared.data(), &saved);
        std::vector<float> attended(hidden.size());
        draft_attention(draft, prefix, context_hidden.data(), 1, prepared.data(), block, attended.data());
        std::vector<float> base(static_cast<std::size_t>(2 * 2 * kDraftHidden));
        draft.load(prefix + "attention_conv.base_kernel", base.data(), static_cast<int>(base.size()));
        std::vector<float> finished(hidden.size());
        grouped_conv(attended.data(), saved.data(), base.data() + 2 * kDraftHidden, block, finished.data());
        for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += finished[i];

        draft.load(prefix + "post_attention_layernorm.weight", norm.data(), kDraftHidden);
        for (int token = 0; token < block; ++token) {
            rmsnorm(hidden.data() + token * kDraftHidden, norm.data(), kDraftHidden, 1e-5f,
                    normed.data() + token * kDraftHidden);
        }
        dynamic_conv(draft, prefix + "mlp_conv.", normed.data(), block, true, prepared.data(), &saved);
        std::vector<float> gate(static_cast<std::size_t>(block * kDraftInter));
        std::vector<float> up(gate.size());
        std::vector<float> mid(gate.size());
        project_rows(draft, prefix + "mlp.gate_proj.weight", prepared.data(), block, kDraftHidden, kDraftInter, gate.data());
        project_rows(draft, prefix + "mlp.up_proj.weight", prepared.data(), block, kDraftHidden, kDraftInter, up.data());
        for (std::size_t i = 0; i < gate.size(); ++i) mid[i] = silu(gate[i]) * up[i];
        std::vector<float> down(hidden.size());
        project_rows(draft, prefix + "mlp.down_proj.weight", mid.data(), block, kDraftInter, kDraftHidden, down.data());
        draft.load(prefix + "mlp_conv.base_kernel", base.data(), static_cast<int>(base.size()));
        std::vector<float> mlp_out(hidden.size());
        grouped_conv(down.data(), saved.data(), base.data() + 2 * kDraftHidden, block, mlp_out.data());
        for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += mlp_out[i];
    }
    std::vector<float> final_norm(kDraftHidden);
    draft.load("norm.weight", final_norm.data(), kDraftHidden);
    for (int token = 0; token < block; ++token) {
        rmsnorm(hidden.data() + token * kDraftHidden, final_norm.data(), kDraftHidden, 1e-5f,
                hidden.data() + token * kDraftHidden);
    }

    std::vector<std::int32_t> path;
    path.reserve(7);
    std::int32_t predecessor = anchor;
    std::vector<float> logits(154880);
    std::vector<float> proj_w(256 * kDraftHidden);
    draft.load("candidate_selector.hidden_projection.weight", proj_w.data(), static_cast<int>(proj_w.size()));
    for (int slot = 1; slot < block; ++slot) {
        target.lm_head_logits(hidden.data() + slot * kDraftHidden, logits.data());
        std::vector<int> cand(16, 0);
        std::vector<float> unary(16, -1e30f);
        std::vector<char> used(logits.size(), 0);
        for (int pick = 0; pick < 16; ++pick) {
            int best = -1;
            for (int id = 0; id < static_cast<int>(logits.size()); ++id) {
                if (used[static_cast<std::size_t>(id)] != 0) continue;
                if (best < 0 || logits[static_cast<std::size_t>(id)] > logits[static_cast<std::size_t>(best)]) best = id;
            }
            used[static_cast<std::size_t>(best)] = 1;
            cand[static_cast<std::size_t>(pick)] = best;
            unary[static_cast<std::size_t>(pick)] = logits[static_cast<std::size_t>(best)];
        }
        std::vector<float> projected(256);
        gemv_bf16(reinterpret_cast<const std::uint16_t*>(draft.get("candidate_selector.hidden_projection.weight").data),
                  hidden.data() + slot * kDraftHidden, 256, kDraftHidden, 0, 256, projected.data());
        auto row = [&](const std::string& name, int id, float* dest) {
            const auto& tensor = draft.get(name);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(tensor.data) + static_cast<std::size_t>(id * 256);
            for (int i = 0; i < 256; ++i) dest[i] = bf16_to_f32(bits[i]);
        };
        std::vector<float> pred(256);
        std::vector<float> succ(256);
        row("candidate_selector.predecessor_codebook", predecessor, pred.data());
        for (int i = 0; i < 256; ++i) pred[i] *= projected[static_cast<std::size_t>(i)];
        int chosen = cand[0];
        float best_score = -1e30f;
        for (int pick = 0; pick < 16; ++pick) {
            row("candidate_selector.successor_codebook", cand[static_cast<std::size_t>(pick)], succ.data());
            float dot = unary[static_cast<std::size_t>(pick)];
            for (int i = 0; i < 256; ++i) dot += pred[static_cast<std::size_t>(i)] * succ[static_cast<std::size_t>(i)];
            if (dot > best_score) {
                best_score = dot;
                chosen = cand[static_cast<std::size_t>(pick)];
            }
        }
        predecessor = chosen;
        path.push_back(chosen);
    }
    (void)proj_w;
    return path;
}

}  // namespace

void swiglu_clamp(const float* gate, const float* up, int n, float limit, float* hidden) {
    for (int i = 0; i < n; ++i) {
        const float capped_gate = std::min(gate[i], limit);
        const float capped_up = std::min(limit, std::max(-limit, up[i]));
        hidden[i] = silu(capped_gate) * capped_up;
    }
}

void router_select(const float* logits, const float* bias, int experts, int top_k, float scaling, bool normalize,
                   int* indices, float* weights) {
    std::vector<float> choice(static_cast<std::size_t>(experts));
    std::vector<float> scores(static_cast<std::size_t>(experts));
    std::vector<char> used(static_cast<std::size_t>(experts), 0);
    for (int expert = 0; expert < experts; ++expert) {
        scores[static_cast<std::size_t>(expert)] = sigmoid(logits[expert]);
        choice[static_cast<std::size_t>(expert)] = scores[static_cast<std::size_t>(expert)] + bias[expert];
    }
    for (int pick = 0; pick < top_k; ++pick) {
        int best = -1;
        for (int expert = 0; expert < experts; ++expert) {
            if (used[static_cast<std::size_t>(expert)] != 0) continue;
            if (best < 0 || choice[static_cast<std::size_t>(expert)] > choice[static_cast<std::size_t>(best)]) best = expert;
        }
        used[static_cast<std::size_t>(best)] = 1;
        indices[pick] = best;
        weights[pick] = scores[static_cast<std::size_t>(best)];
    }
    if (normalize) {
        float sum = 1e-20f;
        for (int pick = 0; pick < top_k; ++pick) sum += weights[pick];
        for (int pick = 0; pick < top_k; ++pick) weights[pick] = weights[pick] / sum * scaling;
    }
}

void kda_forget_gate(const float* proj, const float* dt_bias, const float* a_log, float lower_bound, int heads, int dim,
                     float* g) {
    for (int head = 0; head < heads; ++head) {
        const float decay = std::exp(a_log[head]);
        for (int axis = 0; axis < dim; ++axis) {
            const int index = head * dim + axis;
            g[index] = lower_bound * sigmoid(decay * (proj[index] + dt_bias[index]));
        }
    }
}

void causal_conv_silu(const float* x, const float* weight, float* mem, int channels, int kernel, float* y) {
    if (kernel < 1) fail("causal conv kernel");
    if (x != y && channels > 0) {
        const auto bytes = static_cast<std::uintptr_t>(static_cast<std::size_t>(channels)) * sizeof(float);
        const auto left = reinterpret_cast<std::uintptr_t>(x);
        const auto right = reinterpret_cast<std::uintptr_t>(y);
        if (left < right + bytes && right < left + bytes) fail("causal conv partial overlap");
    }
    const int history = kernel - 1;
    for (int channel = 0; channel < channels; ++channel) {
        const float* taps = weight + static_cast<std::size_t>(channel) * static_cast<std::size_t>(kernel);
        const float raw = x[channel];
        float acc = 0.f;
        if (history > 0) {
            const float* state = mem + static_cast<std::size_t>(channel) * static_cast<std::size_t>(history);
            for (int tap = 0; tap < history; ++tap) acc += taps[tap] * state[tap];
        }
        acc += taps[history] * raw;
        y[channel] = silu(acc);
        if (history <= 0) continue;
        float* state = mem + static_cast<std::size_t>(channel) * static_cast<std::size_t>(history);
        for (int tap = 0; tap < history - 1; ++tap) state[tap] = state[tap + 1];
        // The serving conv cache is bf16. The next token reads this raw value.
        state[history - 1] = bf16_to_f32(f32_to_bf16(raw));
    }
}

void kda_recurrent_heads(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                         int heads, int dim, float* out) {
    const float scale = 1.f / std::sqrt(static_cast<float>(dim));
    for (int head = 0; head < heads; ++head) {
        const int base = head * dim;
        float q_square = 1e-6f;
        float k_square = 1e-6f;
        for (int axis = 0; axis < dim; ++axis) {
            q_square += q[base + axis] * q[base + axis];
            k_square += k[base + axis] * k[base + axis];
        }
        const float q_inv = scale / std::sqrt(q_square);
        const float k_inv = 1.f / std::sqrt(k_square);
        float* matrix = state + static_cast<std::size_t>(head) * static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim);
        std::vector<float> query(static_cast<std::size_t>(dim));
        std::vector<float> key(static_cast<std::size_t>(dim));
        std::vector<float> decay(static_cast<std::size_t>(dim));
        for (int axis = 0; axis < dim; ++axis) {
            query[static_cast<std::size_t>(axis)] = q[base + axis] * q_inv;
            key[static_cast<std::size_t>(axis)] = k[base + axis] * k_inv;
            decay[static_cast<std::size_t>(axis)] = std::exp(g[base + axis]);
            for (int value = 0; value < dim; ++value) {
                matrix[static_cast<std::size_t>(axis * dim + value)] *= decay[static_cast<std::size_t>(axis)];
            }
        }
        std::vector<float> memory(static_cast<std::size_t>(dim), 0.f);
        for (int axis = 0; axis < dim; ++axis) {
            for (int value = 0; value < dim; ++value) {
                memory[static_cast<std::size_t>(value)] += matrix[static_cast<std::size_t>(axis * dim + value)] * key[static_cast<std::size_t>(axis)];
            }
        }
        const float gate = beta[head];
        for (int axis = 0; axis < dim; ++axis) {
            const float delta = (v[base + axis] - memory[static_cast<std::size_t>(axis)]) * gate;
            for (int key_axis = 0; key_axis < dim; ++key_axis) {
                matrix[static_cast<std::size_t>(key_axis * dim + axis)] += key[static_cast<std::size_t>(key_axis)] * delta;
            }
        }
        for (int value = 0; value < dim; ++value) {
            float acc = 0.f;
            for (int axis = 0; axis < dim; ++axis) {
                acc += matrix[static_cast<std::size_t>(axis * dim + value)] * query[static_cast<std::size_t>(axis)];
            }
            out[base + value] = acc;
        }
    }
}

void mhc_project(const float* streams, int hc, int hidden, const float* fn, const float* base, const float* scale,
                 float rms_eps, float hc_eps, int sinkhorn_iters, float* post, float* comb, float* collapsed) {
    const int coeff = (hc + 2) * hc;
    const int flat_n = hc * hidden;
    // The mHC kernel dots the raw streams, then multiplies by rsqrt.
    // Scaling each lane before the dot flips a few bf16 norm lanes.
    double square = 0.0;
    for (int col = 0; col < flat_n; ++col) square += static_cast<double>(streams[col]) * streams[col];
    const float inv = 1.f / std::sqrt(static_cast<float>(square / static_cast<double>(flat_n)) + rms_eps);
    std::vector<float> mixed(static_cast<std::size_t>(coeff));
    for (int row = 0; row < coeff; ++row) {
        const float* weight = fn + static_cast<std::size_t>(row) * static_cast<std::size_t>(flat_n);
        double acc = 0.0;
        for (int col = 0; col < flat_n; ++col) {
            acc += static_cast<double>(weight[col]) * static_cast<double>(streams[col]);
        }
        mixed[static_cast<std::size_t>(row)] = static_cast<float>(acc) * inv;
    }
    std::vector<float> pre(static_cast<std::size_t>(hc));
    for (int stream = 0; stream < hc; ++stream) {
        pre[static_cast<std::size_t>(stream)] =
            sigmoid(mixed[static_cast<std::size_t>(stream)] * scale[0] + base[stream]) + hc_eps;
        post[stream] = 2.f * sigmoid(mixed[static_cast<std::size_t>(hc + stream)] * scale[1] + base[hc + stream]);
    }
    for (int row = 0; row < hc; ++row) {
        float max_logit = -1e30f;
        for (int col = 0; col < hc; ++col) {
            const float logit = mixed[static_cast<std::size_t>(2 * hc + row * hc + col)] * scale[2] +
                                base[2 * hc + row * hc + col];
            comb[row * hc + col] = logit;
            max_logit = std::max(max_logit, logit);
        }
        float sum = 0.f;
        for (int col = 0; col < hc; ++col) {
            comb[row * hc + col] = std::exp(comb[row * hc + col] - max_logit);
            sum += comb[row * hc + col];
        }
        for (int col = 0; col < hc; ++col) comb[row * hc + col] = comb[row * hc + col] / sum + hc_eps;
    }
    auto normalize_columns = [&]() {
        for (int col = 0; col < hc; ++col) {
            float sum = hc_eps;
            for (int row = 0; row < hc; ++row) sum += comb[row * hc + col];
            for (int row = 0; row < hc; ++row) comb[row * hc + col] /= sum;
        }
    };
    auto normalize_rows = [&]() {
        for (int row = 0; row < hc; ++row) {
            float sum = hc_eps;
            for (int col = 0; col < hc; ++col) sum += comb[row * hc + col];
            for (int col = 0; col < hc; ++col) comb[row * hc + col] /= sum;
        }
    };
    normalize_columns();
    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
        normalize_rows();
        normalize_columns();
    }
    for (int dim = 0; dim < hidden; ++dim) {
        float acc = 0.f;
        for (int stream = 0; stream < hc; ++stream) {
            acc += pre[static_cast<std::size_t>(stream)] * streams[stream * hidden + dim];
        }
        collapsed[dim] = acc;
    }
}

void mhc_combine(const float* residual, const float* branch, const float* post, const float* comb, int hc, int hidden,
                 float* out) {
    for (int stream = 0; stream < hc; ++stream) {
        for (int dim = 0; dim < hidden; ++dim) {
            float acc = post[stream] * branch[dim];
            for (int source = 0; source < hc; ++source) {
                acc += comb[source * hc + stream] * residual[source * hidden + dim];
            }
            out[stream * hidden + dim] = acc;
        }
    }
}

bool fixed_prompt_tokens(std::string_view prompt_id, std::string& text, std::vector<std::int32_t>& tokens) {
    if (prompt_id != "fixed-text-v1") return false;
    text = "Hi";
    tokens = {13041};
    return true;
}

namespace {

std::vector<std::int32_t> greedy_tokens(TextModel& model, const std::vector<std::int32_t>& prompt, int new_tokens) {
    model.prime_capture_kda();
    model.set_capture(true);
    for (const auto token : prompt) model.step(token);
    model.set_capture(false);
    std::vector<std::int32_t> generated;
    generated.reserve(static_cast<std::size_t>(new_tokens));
    for (int index = 0; index < new_tokens; ++index) {
        const std::int32_t token = model.argmax();
        generated.push_back(token);
        if (is_generation_eos(token)) break;
        if (index + 1 < new_tokens) model.step(token);
    }
    return generated;
}

std::vector<std::int32_t> dflash_tokens(TextModel& model, const Store& draft, const std::vector<std::int32_t>& prompt,
                                       int new_tokens, std::vector<std::int32_t>& proposals, int& accepted,
                                       std::string& finish, int& verify_rounds) {
    model.prime_capture_kda();
    const DflashOutput output = generate_dflash_continuation(
        model, [&](std::int32_t anchor) { return propose_dflash(model, draft, anchor); }, prompt, new_tokens);
    proposals = output.last_proposals;
    accepted = output.accepted_drafts;
    finish = output.finish;
    verify_rounds = output.verify_rounds;
    return output.tokens;
}

int open_peer_socket(int rank, std::string_view bind_host, std::string_view peer_host, int port) {
    if ((rank != 0 && rank != 1) || port <= 0 || port > 65535) fail("rank link endpoint");
    auto address = [](std::string_view host, int port_num) {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<std::uint16_t>(port_num));
        const std::string text(host);
        if (::inet_pton(AF_INET, text.c_str(), &addr.sin_addr) != 1) fail("rank link address");
        return addr;
    };
    const int buffer = 4 * 1024 * 1024;
    auto enlarge = [&](int fd) {
        ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer));
        ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer));
        const int nodelay = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    };
    if (rank == 0) {
        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) fail("rank link socket");
        const int reuse = 1;
        ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        const sockaddr_in local = address(bind_host, port);
        if (::bind(server, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
            ::close(server);
            fail("rank link bind");
        }
        if (::listen(server, 1) != 0) {
            ::close(server);
            fail("rank link listen");
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
        int accepted = -1;
        while (accepted < 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                ::close(server);
                fail("rank link accept timeout");
            }
            pollfd ready{};
            ready.fd = server;
            ready.events = POLLIN;
            const int polled = ::poll(&ready, 1, 1000);
            if (polled < 0) {
                if (errno == EINTR) continue;
                ::close(server);
                fail("rank link accept failed");
            }
            if (polled == 0) continue;
            accepted = ::accept(server, nullptr, nullptr);
            if (accepted < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                ::close(server);
                fail("rank link accept failed");
            }
        }
        ::close(server);
        enlarge(accepted);
        set_nonblock(accepted);
        return accepted;
    }
    const sockaddr_in local = address(bind_host, 0);
    const sockaddr_in remote = address(peer_host, port);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) fail("rank link socket");
        if (::bind(fd, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
            ::close(fd);
            fail("rank link bind");
        }
        set_nonblock(fd);
        const int connected = ::connect(fd, reinterpret_cast<const sockaddr*>(&remote), sizeof(remote));
        if (connected != 0 && errno != EINPROGRESS) {
            ::close(fd);
            ::poll(nullptr, 0, 200);
            continue;
        }
        if (connected != 0) {
            pollfd ready{};
            ready.fd = fd;
            ready.events = POLLOUT;
            const int polled = ::poll(&ready, 1, 1000);
            int so_error = 0;
            socklen_t length = sizeof(so_error);
            ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &length);
            if (polled <= 0 || so_error != 0) {
                ::close(fd);
                continue;
            }
        }
        enlarge(fd);
        return fd;
    }
    fail("rank link connect timeout");
}

}  // namespace

GenerateResult generate_text_peer(const std::filesystem::path& checkpoint, std::string_view prompt_id, int new_tokens,
                                  int rank, std::string_view bind_host, std::string_view peer_host, int port) {
    GenerateResult result;
    if (!fixed_prompt_tokens(prompt_id, result.prompt_text, result.prompt_tokens)) {
        result.reason = "unknown_prompt";
        return result;
    }
    if (new_tokens < 1 || new_tokens > 8) {
        result.reason = "new_tokens_out_of_range";
        return result;
    }
    SocketFd link(open_peer_socket(rank, bind_host, peer_host, port));
    TextModel model(checkpoint, 1);
    model.set_rank(rank, link.get());
    const std::vector<std::int32_t> generated = greedy_tokens(model, result.prompt_tokens, new_tokens);
    auto send_ids = [&](const std::vector<std::int32_t>& ids) {
        const std::int32_t count = static_cast<std::int32_t>(ids.size());
        send_all(link.get(), &count, sizeof(count));
        if (count > 0) {
            send_all(link.get(), ids.data(), static_cast<std::size_t>(count) * sizeof(std::int32_t));
        }
    };
    auto recv_ids = [&]() {
        std::int32_t count = 0;
        recv_all(link.get(), &count, sizeof(count));
        if (count < 0 || count > new_tokens) fail("peer token count");
        std::vector<std::int32_t> ids(static_cast<std::size_t>(count));
        if (count > 0) recv_all(link.get(), ids.data(), static_cast<std::size_t>(count) * sizeof(std::int32_t));
        return ids;
    };
    std::vector<std::int32_t> other;
    if (rank == 0) {
        send_ids(generated);
        other = recv_ids();
    } else {
        other = recv_ids();
        send_ids(generated);
    }
    result.world_size = 2;
    result.token_ids = generated;
    result.committed = static_cast<std::uint32_t>(generated.size());
    result.logit_margin = model.margin();
    if (rank == 0) {
        result.rank0_tokens = generated;
        result.rank0_committed = result.committed;
        result.rank1_tokens = std::move(other);
        result.rank1_committed = static_cast<std::uint32_t>(result.rank1_tokens.size());
    } else {
        result.rank1_tokens = generated;
        result.rank1_committed = result.committed;
        result.rank0_tokens = std::move(other);
        result.rank0_committed = static_cast<std::uint32_t>(result.rank0_tokens.size());
    }
    result.ok = result.rank0_tokens == result.rank1_tokens && !result.rank0_tokens.empty();
    if (!result.ok) result.reason = "tp2_rank_mismatch";
    return result;
}

GenerateResult generate_text(const std::filesystem::path& checkpoint, std::string_view prompt_id, std::string_view mode,
                             int new_tokens, const std::filesystem::path& draft_checkpoint) {
    GenerateResult result;
    if (!fixed_prompt_tokens(prompt_id, result.prompt_text, result.prompt_tokens)) {
        result.reason = "unknown_prompt";
        return result;
    }
    if (mode != "greedy" && mode != "tp2" && mode != "dflash") {
        result.reason = "unknown_mode";
        return result;
    }
    if (new_tokens < 1 || new_tokens > 8) {
        result.reason = "new_tokens_out_of_range";
        return result;
    }
    const bool paired = mode == "tp2" || mode == "dflash";
    result.world_size = paired ? 2 : 1;
    auto finish = [&](TextModel& model, std::vector<std::int32_t> generated) {
        result.logit_margin = model.margin();
        result.token_ids = std::move(generated);
        result.committed = static_cast<std::uint32_t>(result.token_ids.size());
        result.ok = true;
    };
    if (paired) {
        if (mode == "dflash" && draft_checkpoint.empty()) {
            result.reason = "draft_absent";
            return result;
        }
        int raw_fds[2] = {-1, -1};
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, raw_fds) != 0) {
            result.reason = "socketpair_failed";
            return result;
        }
        SocketFd parent_end(raw_fds[0]);
        SocketFd child_end(raw_fds[1]);
        const int buffer = 4 * 1024 * 1024;
        ::setsockopt(parent_end.get(), SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer));
        ::setsockopt(parent_end.get(), SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer));
        ::setsockopt(child_end.get(), SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer));
        ::setsockopt(child_end.get(), SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer));
        set_nonblock(parent_end.get());
        set_nonblock(child_end.get());
        std::string dflash_finish;
        int verify_rounds = 0;
        const pid_t pid = ::fork();
        if (pid < 0) {
            result.reason = "fork_failed";
            return result;
        }
        auto produce = [&](TextModel& model, std::vector<std::int32_t>& produced, std::vector<std::int32_t>& produced_drafts,
                           int& produced_accept) {
            if (mode == "dflash") {
                Store draft(draft_checkpoint);
                produced = dflash_tokens(model, draft, result.prompt_tokens, new_tokens, produced_drafts, produced_accept,
                                         dflash_finish, verify_rounds);
            } else {
                produced = greedy_tokens(model, result.prompt_tokens, new_tokens);
            }
        };
        if (pid == 0) {
            parent_end.reset();
            int exit_code = 0;
            try {
                TextModel model(checkpoint, 1);
                model.set_rank(1, child_end.get());
                std::vector<std::int32_t> produced;
                std::vector<std::int32_t> produced_drafts;
                int produced_accept = -1;
                produce(model, produced, produced_drafts, produced_accept);
                const int count = static_cast<int>(produced.size());
                send_all(child_end.get(), &count, sizeof(count));
                if (count > 0) {
                    send_all(child_end.get(), produced.data(), static_cast<std::size_t>(count) * sizeof(std::int32_t));
                }
                send_all(child_end.get(), &produced_accept, sizeof(produced_accept));
                const int draft_count = static_cast<int>(produced_drafts.size());
                send_all(child_end.get(), &draft_count, sizeof(draft_count));
                if (draft_count > 0) {
                    send_all(child_end.get(), produced_drafts.data(),
                             static_cast<std::size_t>(draft_count) * sizeof(std::int32_t));
                }
            } catch (...) {
                exit_code = 1;
                const int count = -1;
                try {
                    send_all(child_end.get(), &count, sizeof(count));
                } catch (...) {
                }
            }
            _exit(exit_code);
        }
        ChildProcess child_proc(pid);
        child_end.reset();
        try {
            TextModel model(checkpoint, 1);
            model.set_rank(0, parent_end.get());
            std::vector<std::int32_t> generated;
            std::vector<std::int32_t> drafts;
            int accepted = -1;
            produce(model, generated, drafts, accepted);
            int child_count = 0;
            recv_all(parent_end.get(), &child_count, sizeof(child_count));
            const bool count_ok = child_count >= 0 && child_count <= new_tokens;
            std::vector<std::int32_t> child;
            int child_accept = -1;
            std::vector<std::int32_t> child_draft;
            bool payload_ok = count_ok;
            if (count_ok) {
                if (child_count > 0) {
                    child.resize(static_cast<std::size_t>(child_count));
                    recv_all(parent_end.get(), child.data(), static_cast<std::size_t>(child_count) * sizeof(std::int32_t));
                }
                recv_all(parent_end.get(), &child_accept, sizeof(child_accept));
                int draft_count = 0;
                recv_all(parent_end.get(), &draft_count, sizeof(draft_count));
                if (draft_count < 0 || draft_count > 7) {
                    payload_ok = false;
                } else if (draft_count > 0) {
                    child_draft.resize(static_cast<std::size_t>(draft_count));
                    recv_all(parent_end.get(), child_draft.data(),
                             static_cast<std::size_t>(draft_count) * sizeof(std::int32_t));
                }
            }
            parent_end.reset();
            const bool child_ok = child_proc.reap();
            RankRecord rank0;
            RankRecord rank1;
            bool agreed = false;
            if (child_ok && payload_ok) {
                const CommitView mine{generated.data(), static_cast<std::uint32_t>(generated.size())};
                const CommitView theirs{child.data(), static_cast<std::uint32_t>(child.size())};
                agreed = commit_same_tokens(mine, theirs, &rank0, &rank1) && drafts == child_draft &&
                         accepted == child_accept;
            }
            if (!agreed) {
                result.ok = false;
                result.reason = std::string(mode) + "_rank_mismatch child_count=" + std::to_string(child_count);
                if (!child_ok) result.reason += " child_status";
                if (!child.empty()) result.reason += " child0=" + std::to_string(child[0]);
                result.token_ids = generated;
                result.rank0_tokens = generated;
                result.rank1_tokens = child;
                result.committed = 0;
                return result;
            }
            if (mode == "dflash") {
                const bool budget_done = dflash_finish == "budget" && static_cast<int>(generated.size()) == new_tokens;
                const bool eos_done = dflash_finish == "eos" && !generated.empty() && is_generation_eos(generated.back());
                if (!budget_done && !eos_done) {
                    result.ok = false;
                    result.reason = dflash_finish.empty() ? "dflash_unfinished" : dflash_finish;
                    result.token_ids = generated;
                    result.rank0_tokens = generated;
                    result.rank1_tokens = child;
                    result.committed = 0;
                    return result;
                }
                result.reason = dflash_finish;
                result.draft_tokens = drafts;
                result.accept_count = accepted;
                if (verify_rounds == 0) result.dflash_accept = "anchor_only";
                else if (accepted == 7 * verify_rounds) result.dflash_accept = "full_plus_bonus";
                else if (accepted == 0) result.dflash_accept = "bonus_only";
                else result.dflash_accept = "partial_plus_bonus";
            }
            finish(model, generated);
            result.rank0_tokens = generated;
            result.rank1_tokens = child;
            result.rank0_committed = rank0.committed;
            result.rank1_committed = rank1.committed;
            return result;
        } catch (...) {
            parent_end.reset();
            child_proc.reap();
            throw;
        }
    }

    TextModel model(checkpoint, 1);
    finish(model, greedy_tokens(model, result.prompt_tokens, new_tokens));
    result.rank0_tokens = result.token_ids;
    result.rank0_committed = result.committed;
    return result;
}

}  // namespace ninfer::glm53
