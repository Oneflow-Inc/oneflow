cmake_minimum_required(VERSION 3.13.4)
include(FetchContent)
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

# llvm build options
set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf
set(LLVM_INCLUDE_EXAMPLES OFF)
set(LLVM_INCLUDE_TESTS OFF)
set(LLVM_INCLUDE_BENCHMARKS OFF)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX")
set(LLVM_ENABLE_ASSERTIONS ON)
set(LLVM_ENABLE_PROJECTS "mlir")
set(LLVM_APPEND_VC_REV OFF)

set(ONEFLOW_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

FetchContent_Declare(
  llvm_monorepo
  FETCHCONTENT_QUIET FALSE
  USES_TERMINAL_DOWNLOAD TRUE
  URL "https://github.com/llvm/llvm-project/archive/a4e2a5145a29af678139f33e94ab3df0fc973e59.zip"
  URL_HASH MD5=d501547c39d344c6f52b5c2e757c3953
)
FetchContent_GetProperties(llvm_monorepo)
if(NOT llvm_monorepo_POPULATED)
  FetchContent_Populate(llvm_monorepo)
  SET(LLVM_ROOT ${llvm_monorepo_SOURCE_DIR})
  set(LLVM_BINARY_DIR ${llvm_monorepo_BINARY_DIR})
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_BINARY_DIR}/lib)
  set(LLVM_EXTERNAL_LIT=${LLVM_BINARY_DIR}/bin/llvm-lit)
  add_subdirectory(${LLVM_ROOT}/llvm ${LLVM_BINARY_DIR})
  set(LLVM_CMAKE_DIR ${LLVM_ROOT}/llvm/cmake/modules)
  set(MLIR_CMAKE_DIR ${LLVM_ROOT}/mlir/cmake/modules)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  # set MLIR_TABLEGEN_EXE for functions in AddMLIR
  set(MLIR_TABLEGEN_EXE mlir-tblgen)
  # set LLVM_TOOLS_BINARY_DIR for function `configure_lit_site_cfg`
  set(LLVM_TOOLS_BINARY_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(LLVM_INCLUDE_DIRS
      ${LLVM_ROOT}/llvm/include
      ${LLVM_BINARY_DIR}/include
    )
  set(MLIR_INCLUDE_DIRS
      ${LLVM_ROOT}/mlir/include
      ${LLVM_BINARY_DIR}/tools/mlir/include
      )
  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(oneflow-opt)
add_subdirectory(oneflow-translate)
