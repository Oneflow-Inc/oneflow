cmake_minimum_required(VERSION 3.5)

project(oneflow)

set(CMAKE_CXX_STANDARD 11)

# TODO: auto generate by foreach
add_executable(balanced_splitter_test_exe ${PROJECT_SOURCE_DIR}/oneflow/common/balanced_splitter.cpp ${PROJECT_SOURCE_DIR}/oneflow/common/balanced_splitter_test.cpp)
add_test(NAME balanced_splitter_test COMMAND balanced_splitter_test_exe)
