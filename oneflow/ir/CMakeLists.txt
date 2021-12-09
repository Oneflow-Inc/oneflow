cmake_minimum_required(VERSION 3.13.4)
# prevent LLVM_DEFINITIONS has a TRUE in it
unset(result CACHE)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(oneflow-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS} PARENT_SCOPE)
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} PARENT_SCOPE)
set(ONEFLOW_MLIR_SOURCE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include PARENT_SCOPE)
set(ONEFLOW_MLIR_BINARY_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/include PARENT_SCOPE)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set_property(GLOBAL PROPERTY ALL_ONEFLOW_LIBS
  -Wl,--no-as-needed oneflow -Wl,--as-needed
  -Wl,--no-as-needed ${oneflow_exe_third_party_libs} -Wl,--as-needed
  -Wl,--no-as-needed ${oneflow_third_party_libs} -Wl,--as-needed
)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(oneflow-opt)
add_subdirectory(oneflow-translate)
add_subdirectory(oneflow-extension)
