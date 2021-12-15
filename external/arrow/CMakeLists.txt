include(FetchContent)
set(ARROW_URL https://github.com/apache/arrow/archive/refs/tags/apache-arrow-7.0.0.dev.tar.gz)
set(ARROW_MD5)
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
set(CMAKE_CXX_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)

set(llvm_monorepo_BINARY_DIR ${llvm_monorepo_BINARY_DIR})
install(TARGETS oneflow of_protoobj of_cfgobj of_functional_obj EXPORT oneflow DESTINATION lib)
install(EXPORT oneflow DESTINATION lib/oneflow)
add_subdirectory(${llvm_monorepo_SOURCE_DIR}/llvm ${llvm_monorepo_BINARY_DIR})
set(LLVM_INCLUDE_DIRS ${LLVM_MAIN_SRC_DIR}/include;${llvm_monorepo_BINARY_DIR}/include)
set(LLVM_EXTERNAL_LIT "${llvm_monorepo_BINARY_DIR}/bin/llvm-lit" CACHE STRING "" FORCE)
set(LTDL_SHLIB_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(LLVM_LIBRARY_DIR "${llvm_monorepo_BINARY_DIR}/lib")
