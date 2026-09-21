#pragma once

#include <cctype>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::glm53::json_scan {

class Cursor {
public:
    explicit Cursor(std::string_view input, std::size_t start = 0) : in_(input), i_(start) {}

    [[nodiscard]] std::size_t position() const noexcept { return i_; }

    void skip_ws() {
        while (i_ < in_.size() && std::isspace(static_cast<unsigned char>(in_[i_])) != 0) ++i_;
    }

    [[nodiscard]] char peek() {
        skip_ws();
        if (i_ >= in_.size()) fail("unexpected end");
        return in_[i_];
    }

    char get() {
        const char ch = peek();
        ++i_;
        return ch;
    }

    void expect(char ch) {
        if (get() != ch) fail("expected delimiter");
    }

    [[nodiscard]] std::string parse_string() {
        if (get() != '"') fail("expected string");
        std::string out;
        while (i_ < in_.size()) {
            const char ch = in_[i_++];
            if (ch == '"') return out;
            if (ch != '\\') {
                out.push_back(ch);
                continue;
            }
            if (i_ >= in_.size()) fail("truncated escape");
            const char escaped = in_[i_++];
            switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(escaped);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u':
                    append_unicode(out);
                    break;
                default:
                    fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    [[nodiscard]] std::int64_t parse_int() {
        skip_ws();
        if (i_ >= in_.size()) fail("expected integer");
        bool neg = false;
        if (in_[i_] == '-') {
            neg = true;
            ++i_;
        }
        if (i_ >= in_.size() || std::isdigit(static_cast<unsigned char>(in_[i_])) == 0) fail("expected integer");
        std::int64_t value = 0;
        while (i_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[i_])) != 0) {
            const int digit = in_[i_] - '0';
            if (value > (INT64_MAX - digit) / 10) fail("integer overflow");
            value = value * 10 + digit;
            ++i_;
        }
        if (i_ < in_.size()) {
            const char next = in_[i_];
            if (next == '.' || next == 'e' || next == 'E') fail("expected integer");
        }
        if (neg) {
            if (value == INT64_MAX) {
                // INT64_MIN cannot be represented as a positive magnitude here; shapes never need it.
                fail("integer overflow");
            }
            value = -value;
        }
        return value;
    }

    [[nodiscard]] bool parse_bool() {
        skip_ws();
        if (in_.substr(i_, 4) == "true") {
            i_ += 4;
            return true;
        }
        if (in_.substr(i_, 5) == "false") {
            i_ += 5;
            return false;
        }
        fail("expected boolean");
    }

    void skip_value() {
        const char ch = peek();
        if (ch == '"') {
            (void)parse_string();
            return;
        }
        if (ch == '{') {
            skip_object();
            return;
        }
        if (ch == '[') {
            skip_array();
            return;
        }
        if (ch == 't' || ch == 'f') {
            (void)parse_bool();
            return;
        }
        if (ch == 'n') {
            skip_ws();
            if (in_.substr(i_, 4) != "null") fail("expected null");
            i_ += 4;
            return;
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            skip_number();
            return;
        }
        fail("expected value");
    }

private:
    std::string_view in_;
    std::size_t i_;

    [[noreturn]] void fail(const char* message) const {
        throw std::runtime_error(std::string("json: ") + message + " at offset " + std::to_string(i_));
    }

    void append_unicode(std::string& out) {
        if (i_ + 4 > in_.size()) fail("truncated unicode escape");
        int code = 0;
        for (int n = 0; n < 4; ++n) {
            const char hex = in_[i_++];
            code <<= 4;
            if (hex >= '0' && hex <= '9') code += hex - '0';
            else if (hex >= 'a' && hex <= 'f') code += hex - 'a' + 10;
            else if (hex >= 'A' && hex <= 'F') code += hex - 'A' + 10;
            else fail("invalid hex");
        }
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    void skip_number() {
        skip_ws();
        if (i_ < in_.size() && in_[i_] == '-') ++i_;
        if (i_ >= in_.size() || std::isdigit(static_cast<unsigned char>(in_[i_])) == 0) fail("expected number");
        while (i_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[i_])) != 0) ++i_;
        if (i_ < in_.size() && in_[i_] == '.') {
            ++i_;
            while (i_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[i_])) != 0) ++i_;
        }
        if (i_ < in_.size() && (in_[i_] == 'e' || in_[i_] == 'E')) {
            ++i_;
            if (i_ < in_.size() && (in_[i_] == '+' || in_[i_] == '-')) ++i_;
            if (i_ >= in_.size() || std::isdigit(static_cast<unsigned char>(in_[i_])) == 0) fail("expected exponent");
            while (i_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[i_])) != 0) ++i_;
        }
    }

    void skip_object() {
        expect('{');
        skip_ws();
        if (peek() == '}') {
            ++i_;
            return;
        }
        while (true) {
            (void)parse_string();
            expect(':');
            skip_value();
            const char ch = peek();
            if (ch == ',') {
                ++i_;
                continue;
            }
            if (ch == '}') {
                ++i_;
                return;
            }
            fail("expected ',' or '}'");
        }
    }

    void skip_array() {
        expect('[');
        skip_ws();
        if (peek() == ']') {
            ++i_;
            return;
        }
        while (true) {
            skip_value();
            const char ch = peek();
            if (ch == ',') {
                ++i_;
                continue;
            }
            if (ch == ']') {
                ++i_;
                return;
            }
            fail("expected ',' or ']'");
        }
    }
};

inline std::size_t find_key_value(std::string_view text, std::string_view key) {
    const std::string pattern = "\"" + std::string(key) + "\"";
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto found = text.find(pattern, pos);
        if (found == std::string_view::npos) return std::string_view::npos;
        std::size_t cursor = found + pattern.size();
        while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor])) != 0) ++cursor;
        if (cursor < text.size() && text[cursor] == ':') return cursor + 1;
        pos = found + pattern.size();
    }
    return std::string_view::npos;
}

inline std::optional<std::string> find_string(std::string_view text, std::string_view key) {
    const auto at = find_key_value(text, key);
    if (at == std::string_view::npos) return std::nullopt;
    Cursor cursor(text, at);
    if (cursor.peek() != '"') return std::nullopt;
    return cursor.parse_string();
}

inline std::optional<std::int64_t> find_int(std::string_view text, std::string_view key) {
    const auto at = find_key_value(text, key);
    if (at == std::string_view::npos) return std::nullopt;
    Cursor cursor(text, at);
    const char ch = cursor.peek();
    if (ch != '-' && std::isdigit(static_cast<unsigned char>(ch)) == 0) return std::nullopt;
    return cursor.parse_int();
}

inline std::optional<bool> find_bool(std::string_view text, std::string_view key) {
    const auto at = find_key_value(text, key);
    if (at == std::string_view::npos) return std::nullopt;
    Cursor cursor(text, at);
    const char ch = cursor.peek();
    if (ch != 't' && ch != 'f') return std::nullopt;
    return cursor.parse_bool();
}

inline std::optional<std::vector<std::int64_t>> find_int_array(std::string_view text, std::string_view key) {
    const auto at = find_key_value(text, key);
    if (at == std::string_view::npos) return std::nullopt;
    Cursor cursor(text, at);
    if (cursor.peek() != '[') return std::nullopt;
    cursor.expect('[');
    std::vector<std::int64_t> values;
    if (cursor.peek() == ']') {
        cursor.expect(']');
        return values;
    }
    while (true) {
        values.push_back(cursor.parse_int());
        const char ch = cursor.peek();
        if (ch == ',') {
            cursor.expect(',');
            continue;
        }
        if (ch == ']') {
            cursor.expect(']');
            return values;
        }
        return std::nullopt;
    }
}

}  // namespace ninfer::glm53::json_scan
