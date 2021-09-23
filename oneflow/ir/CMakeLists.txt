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

message("-- LLVM_MONO_REPO_URL: " ${LLVM_MONO_REPO_URL})
message("-- LLVM_MONO_REPO_MD5: " ${LLVM_MONO_REPO_MD5})
FetchContent_Declare(
  llvm_monorepo
)
FetchContent_GetProperties(llvm_monorepo)
if(NOT llvm_monorepo_POPULATED)
  FetchContent_Populate(llvm_monorepo
    URL ${LLVM_MONO_REPO_URL}
    URL_HASH MD5=${LLVM_MONO_REPO_MD5}
  )
  SET(LLVM_ROOT ${llvm_monorepo_SOURCE_DIR})
  set(LLVM_BINARY_DIR ${llvm_monorepo_BINARY_DIR})
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_BINARY_DIR}/lib)
  set(LLVM_INSTALL_DIR ${THIRD_PARTY_DIR}/llvm)

  execute_process(COMMAND "${CMAKE_COMMAND}" ${llvm_monorepo_SOURCE_DIR}/llvm
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
    -DLLVM_ENABLE_RTTI=ON # turn this on to make it compatible with protobuf
    -DLLVM_ENABLE_EH=ON # turn this on to make it compatible with half (the library)
    -DLLVM_INCLUDE_EXAMPLES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_TARGETS_TO_BUILD=host\;NVPTX
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_APPEND_VC_REV=OFF
    -DLLVM_ENABLE_ZLIB=OFF
    -DLLVM_INSTALL_UTILS=ON
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DLLVM_ENABLE_OCAMLDOC=OFF
    -DLLVM_ENABLE_BINDINGS=OFF
    -DMLIR_ENABLE_CUDA_RUNNER=${BUILD_CUDA}
    -DCMAKE_CUDA_COMPILER=${CUDA_NVCC_EXECUTABLE}
    -G ${CMAKE_GENERATOR}
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
endif()

set(LLVM_EXTERNAL_LIT "${LLVM_BINARY_DIR}/bin/llvm-lit")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(oneflow-opt)
add_subdirectory(oneflow-translate)
add_subdirectory(oneflow-extension)
