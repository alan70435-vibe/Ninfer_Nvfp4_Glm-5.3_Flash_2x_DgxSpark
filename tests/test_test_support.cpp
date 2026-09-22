#include "test_support.hpp"
#include <string_view>

int main(int argc, char** argv) {
    int evaluations = 0;
    CHECK(++evaluations == 1);
    CHECK(evaluations == 1);
    if (argc == 2 && std::string_view(argv[1]) == "--fail") CHECK(false);
    CHECK(argc == 1);
    return 0;
}
