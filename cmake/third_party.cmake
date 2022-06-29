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
include(lz4)
include(string_view)

include(hwloc)
include(liburing)
if(WITH_ONEDNN)
  include(oneDNN)
endif()

set_mirror_url_with_hash(INJA_URL https://github.com/pantor/inja/archive/refs/tags/v3.3.0.zip
                         611e6b7206d0fb89728a3879f78b4775)

option(CUDA_STATIC "" ON)

if(BUILD_CUDA)
  if((NOT CUDA_STATIC) OR BUILD_SHARED_LIBS)
    set(OF_CUDA_LINK_DYNAMIC_LIBRARY ON)
  else()
    set(OF_CUDA_LINK_DYNAMIC_LIBRARY OFF)
  endif()
  if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
    message(WARNING "CUDA_TOOLKIT_ROOT_DIR is deprecated, use CUDAToolkit_ROOT instead")
    set(CUDAToolkit_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
  endif(DEFINED CUDA_TOOLKIT_ROOT_DIR)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
  message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDAToolkit_VERSION_MAJOR: ${CUDAToolkit_VERSION_MAJOR}")
  message(STATUS "CUDAToolkit_VERSION_MINOR: ${CUDAToolkit_VERSION_MINOR}")
  message(STATUS "CUDAToolkit_VERSION_PATCH: ${CUDAToolkit_VERSION_PATCH}")
  message(STATUS "CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
  message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
  message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
  message(STATUS "CUDAToolkit_LIBRARY_ROOT: ${CUDAToolkit_LIBRARY_ROOT}")
  message(STATUS "CUDAToolkit_TARGET_DIR: ${CUDAToolkit_TARGET_DIR}")
  message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")
  if(CUDA_NVCC_GENCODES)
    message(FATAL_ERROR "CUDA_NVCC_GENCODES is deprecated, use CMAKE_CUDA_ARCHITECTURES instead")
  endif()
  add_definitions(-DWITH_CUDA)
  # NOTE: For some unknown reason, CUDAToolkit_VERSION may become empty when running cmake again
  set(CUDA_VERSION ${CUDAToolkit_VERSION} CACHE STRING "")
  if(NOT CUDA_VERSION)
    message(FATAL_ERROR "CUDA_VERSION empty")
  endif()
  message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
  if(OF_CUDA_LINK_DYNAMIC_LIBRARY)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cublas)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::curand)
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.1")
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cublasLt)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.2")
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::nvjpeg)
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::nppc)
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::nppig)
    endif()
  else()
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cublas_static)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::curand_static)
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.1")
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cublasLt_static)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.2")
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::nvjpeg_static)
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::nppig_static)
      # Must put nppc_static after nppig_static in CUDA 10.2
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::nppc_static)
      list(APPEND VENDOR_CUDA_LIBRARIES CUDA::culibos)
    endif()
  endif()
  message(STATUS "VENDOR_CUDA_LIBRARIES: ${VENDOR_CUDA_LIBRARIES}")
  # add a cache entry if want to use a ccache/sccache wrapped nvcc
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE} CACHE STRING "")
  message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_STANDARD 14)
  find_package(CUDNN REQUIRED)
endif()

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
    ${LZ4_STATIC_LIBRARIES}
    nlohmann_json::nlohmann_json
    string-view-lite)
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
    flatbuffers
    lz4_copy_libs_to_destination
    lz4_copy_headers_to_destination)
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
  ${FLATBUFFERS_INCLUDE_DIR}
  ${LZ4_INCLUDE_DIR})
if(WITH_ONEDNN)
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${ONEDNN_INCLUDE_DIR})
endif()

list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${RE2_INCLUDE_DIR})

if(BUILD_CUDA)
  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
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

  list(APPEND oneflow_third_party_libs ${NCCL_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${CUDNN_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${VENDOR_CUDA_LIBRARIES})

  list(APPEND oneflow_third_party_dependencies nccl)

  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${CUDNN_INCLUDE_DIRS} ${CUB_INCLUDE_DIR}
       ${NCCL_INCLUDE_DIR})
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

if(WITH_LIBURING)
  list(APPEND oneflow_third_party_dependencies liburing)
  list(APPEND oneflow_third_party_libs ${LIBURING_STATIC_LIBRARIES})
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${LIBURING_INCLUDE_DIR})
  add_definitions(-DWITH_LIBURING)
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
