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

set(LLVM_MONO_REPO_URL "https://github.com/llvm/llvm-project/archive/a4e2a5145a29af678139f33e94ab3df0fc973e59.zip" CACHE STRING "")
set(LLVM_MONO_REPO_MD5 "d501547c39d344c6f52b5c2e757c3953" CACHE STRING "")
message("-- LLVM_MONO_REPO_URL: " ${LLVM_MONO_REPO_URL})
message("-- LLVM_MONO_REPO_MD5: " ${LLVM_MONO_REPO_MD5})
FetchContent_Declare(
  llvm_monorepo
  FETCHCONTENT_QUIET FALSE
  USES_TERMINAL_DOWNLOAD TRUE
  URL ${LLVM_MONO_REPO_URL}
  URL_HASH MD5=${LLVM_MONO_REPO_MD5}
)
FetchContent_GetProperties(llvm_monorepo)
if(NOT llvm_monorepo_POPULATED)
  FetchContent_Populate(llvm_monorepo)
  SET(LLVM_ROOT ${llvm_monorepo_SOURCE_DIR})
  set(LLVM_BINARY_DIR ${llvm_monorepo_BINARY_DIR})
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_BINARY_DIR}/lib)
  set(LLVM_INSTALL_DIR ${LLVM_BINARY_DIR}/install)
  set(LLVM_EXTERNAL_LIT=${LLVM_BINARY_DIR}/bin/llvm-lit)

  execute_process(COMMAND "${CMAKE_COMMAND}" ${llvm_monorepo_SOURCE_DIR}/llvm
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
    -DLLVM_ENABLE_RTTI=ON # turn this on to make it compatible with protobuf
    -DLLVM_INCLUDE_EXAMPLES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_TARGETS_TO_BUILD=X86\;NVPTX
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_APPEND_VC_REV=OFF
    -DLLVM_ENABLE_ZLIB=OFF
    -DLLVM_INSTALL_UTILS=ON
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    WORKING_DIRECTORY ${llvm_monorepo_BINARY_DIR}
    RESULT_VARIABLE ret)
  if(ret EQUAL "1")
      message( FATAL_ERROR "Bad exit status")
  endif()
  include(ProcessorCount)
  ProcessorCount(PROC_NUM)
  execute_process(COMMAND "${CMAKE_COMMAND}" --build . -j${PROC_NUM}
    WORKING_DIRECTORY ${llvm_monorepo_BINARY_DIR}
    RESULT_VARIABLE ret
  )
  if(ret EQUAL "1")
      message( FATAL_ERROR "Bad exit status")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" --build . -j${PROC_NUM} --target install
    WORKING_DIRECTORY ${llvm_monorepo_BINARY_DIR}
    RESULT_VARIABLE ret
  )
  if(ret EQUAL "1")
      message( FATAL_ERROR "Bad exit status")
  endif()
  set(LLVM_DIR ${LLVM_INSTALL_DIR}/lib/cmake/llvm)
  set(MLIR_DIR ${LLVM_INSTALL_DIR}/lib/cmake/mlir)
  find_package(MLIR REQUIRED CONFIG)

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
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
  set(LLVM_EXTERNAL_LIT=${llvm_monorepo_BINARY_DIR}/bin/llvm-lit)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(oneflow-opt)
add_subdirectory(oneflow-translate)
