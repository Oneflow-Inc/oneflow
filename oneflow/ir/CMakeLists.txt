cmake_minimum_required(VERSION 3.13.4)
include(FetchContent)
# prevent LLVM_DEFINITIONS has a TRUE in it
unset(result CACHE)
set(CMAKE_INSTALL_MESSAGE LAZY)
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

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(oneflow-dialect LANGUAGES CXX C)

# https://github.com/llvm/llvm-project/issues/55010
set(LLVM_ABI_BREAKING_CHECKS "FORCE_OFF" CACHE STRING "")

if(LLVM_PROVIDER STREQUAL "in-tree")
  include(llvm-in-tree.cmake)
elseif(LLVM_PROVIDER STREQUAL "install")
  include(install-llvm.cmake)
else()
  message(FATAL_ERROR "LLVM_PROVIDER should be in-tree or install, but got: ${LLVM_PROVIDER}")
endif()

set_property(GLOBAL PROPERTY LLVM_INSTALL_DIR ${LLVM_INSTALL_DIR})
set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(MLIR_PDLL_TABLEGEN_EXE mlir-pdll)

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

if(LLVM_PROVIDER STREQUAL "in-tree")
  add_subdirectory(${CMAKE_SOURCE_DIR}/tools/oneflow-tblgen ${PROJECT_BINARY_DIR}/oneflow-tblgen)
endif()

function(update_rpath)
  set_property(TARGET ${ARGV0} APPEND PROPERTY BUILD_RPATH "${LLVM_LIBRARY_DIR}")
  set_property(TARGET ${ARGV0} APPEND PROPERTY BUILD_RPATH "${ONEFLOW_BUILD_ROOT_DIR}")
  set_property(TARGET ${ARGV0} APPEND PROPERTY INSTALL_RPATH "${LLVM_LIBRARY_DIR}")
  set_property(TARGET ${ARGV0} APPEND PROPERTY INSTALL_RPATH "${ONEFLOW_BUILD_ROOT_DIR}")
endfunction(update_rpath)

function(oneflow_add_mlir_library)
  add_mlir_library(${ARGV})
  set_compile_options_to_oneflow_target(${ARGV0})
  update_rpath(${ARGV0})
endfunction()

function(oneflow_add_mlir_dialect_library)
  add_mlir_dialect_library(${ARGV})
  set_compile_options_to_oneflow_target(${ARGV0})
  update_rpath(${ARGV0})
endfunction()

function(oneflow_add_llvm_tool)
  add_llvm_tool(${ARGV})
  llvm_update_compile_flags(oneflow-runner)
  set_compile_options_to_oneflow_target(${ARGV0})
  update_rpath(${ARGV0})
endfunction()

find_package(Threads REQUIRED)
set(LLVM_PTHREAD_LIB ${CMAKE_THREAD_LIBS_INIT})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/lib)
if(WITH_MLIR)
  add_subdirectory(include)
  add_subdirectory(lib)
  add_subdirectory(test)
  add_subdirectory(oneflow-translate)
  add_subdirectory(oneflow-runtime)
  add_subdirectory(oneflow-extension)
  add_subdirectory(oneflow-opt)
  add_subdirectory(oneflow-runner)
  add_subdirectory(oneflow-lite)
endif(WITH_MLIR)

if(BUILD_PYTHON)
  foreach(llvm_include_dir ${LLVM_INCLUDE_DIRS})
    if(llvm_include_dir MATCHES "/include$")
      list(APPEND LLVM_INSTALL_INCLUDE_DIRS "${llvm_include_dir}//")
    else()
      list(APPEND LLVM_INSTALL_INCLUDE_DIRS "${llvm_include_dir}")
    endif()
  endforeach()
  install(
    DIRECTORY ${LLVM_INSTALL_INCLUDE_DIRS}
    DESTINATION ${ONEFLOW_INCLUDE_DIR}
    COMPONENT oneflow_py_include
    EXCLUDE_FROM_ALL FILES_MATCHING
    PATTERN llvm/ADT/ArrayRef.h
    PATTERN llvm/ADT/Hashing.h
    PATTERN llvm/ADT/iterator.h
    PATTERN llvm/ADT/None.h
    PATTERN llvm/ADT/SmallVector.h
    PATTERN llvm/ADT/STLExtras.h
    PATTERN llvm/ADT/STLFunctionalExtras.h
    PATTERN llvm/ADT/DenseMapInfo.h
    PATTERN llvm/ADT/identity.h
    PATTERN llvm/ADT/iterator_range.h
    PATTERN llvm/ADT/Optional.h
    PATTERN llvm/ADT/STLArrayExtras.h
    PATTERN llvm/ADT/STLForwardCompat.h
    PATTERN llvm/ADT/StringRef.h
    PATTERN llvm/Config/abi-breaking.h
    PATTERN llvm/Config/llvm-config.h
    PATTERN llvm/Support/Compiler.h
    PATTERN llvm/Support/DataTypes.h
    PATTERN llvm/Support/ErrorHandling.h
    PATTERN llvm/Support/SwapByteOrder.h
    PATTERN llvm/Support/type_traits.h
    PATTERN llvm-c/DataTypes.h)
endif()
