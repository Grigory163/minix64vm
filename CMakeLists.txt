cmake_minimum_required(VERSION 3.16)
project(MiniX64Vm LANGUAGES CXX)

option(MINIX64VM_BUILD_TESTS "Build the self-test executable" ON)
option(MINIX64VM_BUILD_EXAMPLES "Build the examples" ON)

add_library(minix64vm STATIC
    src/MiniX64Vm.cpp
    src/MiniX64VmVex.inl
    include/MiniX64Vm.h
)
add_library(MiniX64Vm::minix64vm ALIAS minix64vm)

target_include_directories(minix64vm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(minix64vm PUBLIC cxx_std_17)

# AES instructions are executed with host AES-NI intrinsics.
if(MSVC)
    target_compile_options(minix64vm PRIVATE /W4 /wd4324)
else()
    target_compile_options(minix64vm PRIVATE -Wall -Wextra -maes -msse4.1)
endif()

if(MINIX64VM_BUILD_TESTS)
    enable_testing()
    add_executable(minix64vm_tests tests/main.cpp)
    target_link_libraries(minix64vm_tests PRIVATE minix64vm)
    add_test(NAME minix64vm_tests COMMAND minix64vm_tests)
endif()

if(MINIX64VM_BUILD_EXAMPLES)
    add_executable(minix64vm_example examples/basic.cpp)
    target_link_libraries(minix64vm_example PRIVATE minix64vm)
endif()
