cmake_minimum_required(VERSION 3.28)

project(ninfer_glm53 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NINFER_GLM53_BUILD_TESTS "Build host-side contract tests" ON)

add_library(ninfer_glm53_core STATIC
    src/model_spec.cpp
    src/deployment_contract.cpp
    src/execution_plan.cpp
    src/parameter_schema.cpp
    src/checkpoint_binding.cpp
)

target_include_directories(ninfer_glm53_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(ninfer_glm53_core PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ninfer_glm53_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_executable(ninfer-glm53-plan apps/ninfer_glm53_plan.cpp)
target_link_libraries(ninfer-glm53-plan PRIVATE ninfer_glm53_core)

add_executable(ninfer-glm53-bind apps/ninfer_glm53_bind.cpp)
target_link_libraries(ninfer-glm53-bind PRIVATE ninfer_glm53_core)

if(NINFER_GLM53_BUILD_TESTS)
    include(CTest)
    enable_testing()

    foreach(test_name IN ITEMS model_spec deployment_contract execution_plan checkpoint_binding)
        add_executable(test_${test_name} tests/test_${test_name}.cpp)
        target_link_libraries(test_${test_name} PRIVATE ninfer_glm53_core)
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()
endif()
