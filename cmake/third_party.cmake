cmake_policy(SET CMP0074 NEW)
if(NOT WIN32)
  find_package(Threads)
endif()

if(WITH_ZLIB)
  include(zlib)
endif()
include(protobuf)
include(googletest)
include(glog)
include(libjpeg-turbo)
include(opencv)
include(eigen)
if(WITH_COCOAPI)
  include(cocoapi)
endif()
include(half)
include(re2)
include(json)
if(RPC_BACKEND MATCHES "GRPC")
  include(absl)
  include(cares)
  include(openssl)
  include(grpc)
endif()
include(flatbuffers)

include(hwloc)
if(WITH_ONEDNN)
  include(oneDNN)
endif()

set_mirror_url_with_hash(INJA_URL https://github.com/pantor/inja/archive/refs/tags/v3.3.0.zip
                         611e6b7206d0fb89728a3879f78b4775)

if(NOT WIN32)
  set(BLA_STATIC ON)
  set(BLA_VENDOR "Intel10_64lp_seq")
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    set(BLA_VENDOR "All")
    find_package(BLAS)
  endif()
else()
  set(MKL_LIB_PATH
      "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2017/windows/mkl/lib/intel64_win"
  )
  set(BLAS_LIBRARIES ${MKL_LIB_PATH}/mkl_core_dll.lib ${MKL_LIB_PATH}/mkl_sequential_dll.lib
                     ${MKL_LIB_PATH}/mkl_intel_lp64_dll.lib)
endif()
message(STATUS "Found Blas Lib: " ${BLAS_LIBRARIES})

set(oneflow_test_libs gtest_main)

set(oneflow_third_party_libs
    protobuf_imported
    ${GRPC_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${OPENCV_STATIC_LIBRARIES}
    ${COCOAPI_STATIC_LIBRARIES}
    ${LIBJPEG_STATIC_LIBRARIES}
    ${ABSL_STATIC_LIBRARIES}
    ${OPENSSL_STATIC_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${FLATBUFFERS_STATIC_LIBRARIES}
    nlohmann_json::nlohmann_json)
if(WITH_ONEDNN)
  set(oneflow_third_party_libs ${oneflow_third_party_libs} ${ONEDNN_STATIC_LIBRARIES})
endif()

list(APPEND oneflow_third_party_libs ${RE2_LIBRARIES})

if(WITH_ZLIB)
  list(APPEND oneflow_third_party_libs zlib_imported)
endif()

if(WIN32)
  # static gflags lib requires "PathMatchSpecA" defined in "ShLwApi.Lib"
  list(APPEND oneflow_third_party_libs "ShLwApi.Lib")
  list(APPEND oneflow_third_party_libs "Ws2_32.lib")
endif()

set(oneflow_third_party_dependencies
    protobuf
    eigen
    half_copy_headers_to_destination
    re2
    opencv
    install_libpng_headers
    flatbuffers)
if(WITH_ONEDNN)
  list(APPEND oneflow_third_party_dependencies onednn)
endif()
if(WITH_ZLIB)
  list(APPEND oneflow_third_party_dependencies zlib)
endif()

if(WITH_COCOAPI)
  list(APPEND oneflow_third_party_dependencies cocoapi_copy_headers_to_destination)
  list(APPEND oneflow_third_party_dependencies cocoapi_copy_libs_to_destination)
endif()

if(RPC_BACKEND MATCHES "GRPC")
  list(APPEND oneflow_third_party_dependencies grpc)
endif()

list(
  APPEND
  ONEFLOW_THIRD_PARTY_INCLUDE_DIRS
  ${ZLIB_INCLUDE_DIR}
  ${PROTOBUF_INCLUDE_DIR}
  ${GRPC_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIR}
  ${LIBJPEG_INCLUDE_DIR}
  ${OPENCV_INCLUDE_DIR}
  ${LIBPNG_INCLUDE_DIR}
  ${EIGEN_INCLUDE_DIR}
  ${COCOAPI_INCLUDE_DIR}
  ${HALF_INCLUDE_DIR}
  ${ABSL_INCLUDE_DIR}
  ${OPENSSL_INCLUDE_DIR}
  ${FLATBUFFERS_INCLUDE_DIR})
if(WITH_ONEDNN)
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${ONEDNN_INCLUDE_DIR})
endif()

list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${RE2_INCLUDE_DIR})

if(BUILD_CUDA)
  # Always use third_party/cub for Clang CUDA in case of compatibility issues
  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA" AND CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    if(CMAKE_CXX_STANDARD LESS 14)
      add_definitions(-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)
      add_definitions(-DCUB_IGNORE_DEPRECATED_CPP11)
    endif()
    if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
      add_definitions(-DCUB_IGNORE_DEPRECATED_COMPILER)
    endif()
  else()
    include(cub)
    list(APPEND oneflow_third_party_dependencies cub_copy_headers_to_destination)
  endif()
  include(nccl)
  include(cutlass)
  include(trt_flash_attention)

  list(APPEND oneflow_third_party_libs ${NCCL_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${CUDNN_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${VENDOR_CUDA_LIBRARIES})

  list(APPEND oneflow_third_party_dependencies nccl)

  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${CUDNN_INCLUDE_DIRS} ${CUB_INCLUDE_DIR}
       ${NCCL_INCLUDE_DIR})

  if(WITH_CUTLASS)
    list(APPEND oneflow_third_party_dependencies cutlass)
    list(APPEND oneflow_third_party_dependencies cutlass_copy_examples_to_destination)
    list(APPEND oneflow_third_party_libs ${CUTLASS_LIBRARIES})
    list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${CUTLASS_INCLUDE_DIR})
  endif()
  list(APPEND oneflow_third_party_dependencies trt_flash_attention)
  list(APPEND oneflow_third_party_libs ${TRT_FLASH_ATTENTION_LIBRARIES})
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${TRT_FLASH_ATTENTION_INCLUDE_DIR})
endif()

if(BUILD_RDMA)
  if(UNIX)
    include(CheckIncludeFiles)
    include(CheckLibraryExists)
    check_include_files(infiniband/verbs.h HAVE_VERBS_H)
    if(HAVE_VERBS_H)
      add_definitions(-DWITH_RDMA)
    else()
      message(FATAL_ERROR "RDMA head file not found")
    endif()
  else()
    message(FATAL_ERROR "UNIMPLEMENTED")
  endif()
endif()

if(BUILD_HWLOC)
  list(APPEND oneflow_third_party_dependencies hwloc)
  list(APPEND oneflow_third_party_libs ${ONEFLOW_HWLOC_STATIC_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${PCIACCESS_STATIC_LIBRARIES})
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})
  add_definitions(-DWITH_HWLOC)
endif()

include_directories(SYSTEM ${ONEFLOW_THIRD_PARTY_INCLUDE_DIRS})

foreach(oneflow_third_party_lib IN LISTS oneflow_third_party_libs)
  if(NOT "${oneflow_third_party_lib}" MATCHES "^-l.+"
     AND NOT TARGET ${oneflow_third_party_lib}
     AND "${oneflow_third_party_lib}" MATCHES "^\/.+"
     AND NOT "${oneflow_third_party_lib}" MATCHES "^.+\.framework")
    get_filename_component(IMPORTED_LIB_NAME ${oneflow_third_party_lib} NAME_WE)
    set(IMPORTED_LIB_NAME "imported::${IMPORTED_LIB_NAME}")
    message(STATUS "Creating imported lib: ${oneflow_third_party_lib} => ${IMPORTED_LIB_NAME}")
    add_library(${IMPORTED_LIB_NAME} UNKNOWN IMPORTED)
    set_property(TARGET ${IMPORTED_LIB_NAME} PROPERTY IMPORTED_LOCATION
                                                      "${oneflow_third_party_lib}")
    list(APPEND ONEFLOW_THIRD_PARTY_LIBS_TO_LINK "${IMPORTED_LIB_NAME}")
  else()
    list(APPEND ONEFLOW_THIRD_PARTY_LIBS_TO_LINK "${oneflow_third_party_lib}")
  endif()
endforeach()

set(oneflow_third_party_libs ${ONEFLOW_THIRD_PARTY_LIBS_TO_LINK})
message(STATUS "oneflow_third_party_libs: ${oneflow_third_party_libs}")

add_definitions(-DHALF_ENABLE_CPP11_USER_LITERALS=0)

if(THIRD_PARTY)
  add_custom_target(prepare_oneflow_third_party ALL DEPENDS ${oneflow_third_party_dependencies})
  if(BUILD_PYTHON)
    if(NOT ONEFLOW_INCLUDE_DIR MATCHES "/include$")
      message(
        FATAL_ERROR
          "ONEFLOW_INCLUDE_DIR must end with '/include', current value: ${ONEFLOW_INCLUDE_DIR}")
    endif()
    get_filename_component(ONEFLOW_INCLUDE_DIR_PARENT "${ONEFLOW_INCLUDE_DIR}" DIRECTORY)
    foreach(of_include_src_dir ${ONEFLOW_THIRD_PARTY_INCLUDE_DIRS})
      if(of_include_src_dir MATCHES "/include$")
        # it requires two slashes, but in CMake doc it states only one slash is needed
        set(of_include_src_dir "${of_include_src_dir}//")
      endif()
      install(
        DIRECTORY ${of_include_src_dir}
        DESTINATION ${ONEFLOW_INCLUDE_DIR}
        COMPONENT oneflow_py_include
        EXCLUDE_FROM_ALL)
    endforeach()
  endif(BUILD_PYTHON)
else()
  add_custom_target(prepare_oneflow_third_party ALL)
endif()
