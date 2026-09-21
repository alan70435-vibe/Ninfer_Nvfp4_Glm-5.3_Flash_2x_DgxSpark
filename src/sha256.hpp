#pragma once

#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::glm53::detail {

class Sha256 {
public:
    void update(std::string_view data) {
        for (unsigned char byte : data) {
            add_byte(byte);
            bits_ += 8U;
        }
    }

    [[nodiscard]] std::string hex_digest() {
        const std::uint64_t bit_length = bits_;
        add_byte(0x80U);
        while (block_len_ != 56U) add_byte(0U);
        for (int shift = 56; shift >= 0; shift -= 8) {
            add_byte(static_cast<std::uint8_t>(bit_length >> static_cast<unsigned>(shift)));
        }
        std::string hex;
        hex.resize(64U);
        static constexpr char kDigits[] = "0123456789abcdef";
        for (int word = 0; word < 8; ++word) {
            for (int nibble = 0; nibble < 8; ++nibble) {
                const auto shift = static_cast<unsigned>(28 - nibble * 4);
                hex[static_cast<std::size_t>(word * 8 + nibble)] =
                    kDigits[(state_[word] >> shift) & 0x0FU];
            }
        }
        return hex;
    }

private:
    void add_byte(std::uint8_t byte) {
        block_[block_len_++] = byte;
        if (block_len_ == 64U) {
            compress();
            block_len_ = 0U;
        }
    }

    static std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
        return (value >> bits) | (value << (32U - bits));
    }

    void compress() {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
            0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
            0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
            0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
            0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
            0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
            0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
            0xc67178f2U};

        std::uint32_t words[64];
        for (int i = 0; i < 16; ++i) {
            const auto base = static_cast<std::size_t>(i * 4);
            words[i] = (static_cast<std::uint32_t>(block_[base]) << 24U) |
                       (static_cast<std::uint32_t>(block_[base + 1U]) << 16U) |
                       (static_cast<std::uint32_t>(block_[base + 2U]) << 8U) |
                       static_cast<std::uint32_t>(block_[base + 3U]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(words[i - 15], 7U) ^ rotr(words[i - 15], 18U) ^ (words[i - 15] >> 3U);
            const std::uint32_t s1 = rotr(words[i - 2], 17U) ^ rotr(words[i - 2], 19U) ^ (words[i - 2] >> 10U);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choose + k[i] + words[i];
            const std::uint32_t s0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::uint32_t state_[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::uint64_t bits_{0};
    std::uint8_t block_[64]{};
    std::size_t block_len_{0};
};

inline std::string sha256_hex(std::string_view bytes) {
    Sha256 hasher;
    hasher.update(bytes);
    return hasher.hex_digest();
}

inline std::string sha256_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    Sha256 hasher;
    char buffer[1 << 20];
    while (input) {
        input.read(buffer, sizeof(buffer));
        const auto count = input.gcount();
        if (count > 0) hasher.update(std::string_view(buffer, static_cast<std::size_t>(count)));
    }
    if (input.bad()) throw std::runtime_error("failed while hashing " + path);
    return hasher.hex_digest();
}

inline bool is_hex40(std::string_view value) {
    if (value.size() != 40U) return false;
    for (unsigned char ch : value) {
        if (std::isxdigit(ch) == 0) return false;
    }
    return true;
}

}  // namespace ninfer::glm53::detail
