cmake_minimum_required(VERSION 3.5)

project(oneflow C CXX)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

enable_testing()

include(cmake/dependencies.cmake)

# NOTE: temporary solution
# TODO: auto generate by foreach
add_executable(balanced_splitter_test_exe
               ${PROJECT_SOURCE_DIR}/oneflow/common/balanced_splitter.cpp
               ${PROJECT_SOURCE_DIR}/oneflow/common/balanced_splitter_test.cpp)
target_link_libraries(balanced_splitter_test_exe ${ONEFLOW_LINKER_LIBS})
add_test(NAME balanced_splitter_test COMMAND balanced_splitter_test_exe)
