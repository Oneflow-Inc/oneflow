include(FetchContent)
message("-- LLVM_MONO_REPO_URL: " ${LLVM_MONO_REPO_URL})
message("-- LLVM_MONO_REPO_MD5: " ${LLVM_MONO_REPO_MD5})
FetchContent_Declare(
  llvm_monorepo
)
FetchContent_GetProperties(llvm_monorepo)

set(LLVM_INSTALL_DIR ${THIRD_PARTY_DIR}/llvm)

if(NOT llvm_monorepo_POPULATED)
  FetchContent_Populate(llvm_monorepo
    URL ${LLVM_MONO_REPO_URL}
    URL_HASH MD5=${LLVM_MONO_REPO_MD5}
  )
endif()

set(CMAKE_INSTALL_PREFIX ${LLVM_INSTALL_DIR} CACHE STRING "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "turn this on to make it compatible with protobuf")
set(LLVM_ENABLE_EH ON CACHE BOOL "turn this on to make it compatible with half (the library)")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_TARGETS_TO_BUILD host;NVPTX CACHE STRING "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS mlir CACHE STRING "")
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(MLIR_ENABLE_CUDA_RUNNER ${WITH_MLIR_CUDA_CODEGEN} CACHE STRING "")
set(LLVM_EXTERNAL_PROJECTS OneFlowTableGen CACHE STRING "")
set(LLVM_EXTERNAL_ONEFLOWTABLEGEN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/tools/oneflow-tblgen CACHE STRING "")

add_subdirectory(${llvm_monorepo_SOURCE_DIR}/llvm ${llvm_monorepo_BINARY_DIR})

add_dependencies(oneflow install-oneflow-tblgen)
