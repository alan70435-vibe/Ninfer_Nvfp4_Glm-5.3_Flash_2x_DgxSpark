#pragma once

#include <cstdlib>
#include <iostream>
#include <utility>

namespace ninfer::test {
inline void check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        std::cerr << file << ':' << line << ": CHECK(" << expression << ") failed\n";
        std::exit(EXIT_FAILURE);
    }
}
template <class Exception, class Function>
bool throws(Function&& function) {
    try { std::forward<Function>(function)(); }
    catch (const Exception&) { return true; }
    catch (...) { return false; }
    return false;
}
}  // namespace ninfer::test

// Intentionally independent of NDEBUG. The expression is evaluated once.
#define CHECK(...) ::ninfer::test::check(static_cast<bool>((__VA_ARGS__)), #__VA_ARGS__, __FILE__, __LINE__)
