if (NOT WIN32)
  find_package(Threads)
endif()

include(zlib)
include(protobuf)
include(googletest)
include(gflags)
include(glog)
include(grpc)
if (BUILD_CUDA)
  include(cub)
endif()

if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DWITH_CUDA)
endif()

if (BUILD_CUDNN)
  if (NOT BUILD_CUDA)
    message(FATAL_ERROR "BUILD_CUDNN without BUILD_CUDA")
  endif()
  find_package(CuDNN REQUIRED)
  add_definitions(-DWITH_CUDNN)
endif()

if (NOT WIN32)
  set(BLA_VENDOR "Intel10_64lp_seq")
  find_package(BLAS)
  if (NOT BLAS_FOUND)
    set(BLA_VENDOR "All")
    find_package(BLAS)
  endif()
else()
  set(MKL_LIB_PATH "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2017/windows/mkl/lib/intel64_win")
  set(BLAS_LIBRARIES ${MKL_LIB_PATH}/mkl_core_dll.lib ${MKL_LIB_PATH}/mkl_sequential_dll.lib ${MKL_LIB_PATH}/mkl_intel_lp64_dll.lib)
endif()
message(STATUS "Found Blas Lib: " ${BLAS_LIBRARIES})

set(oneflow_third_party_libs
    ${CMAKE_THREAD_LIBS_INIT}
    ${GLOG_STATIC_LIBRARIES}
    ${GFLAGS_STATIC_LIBRARIES}
    ${GOOGLETEST_STATIC_LIBRARIES}
    ${GOOGLEMOCK_STATIC_LIBRARIES}
    ${PROTOBUF_STATIC_LIBRARIES}
    ${GRPC_STATIC_LIBRARIES}
    ${ZLIB_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

if(WIN32)
  # static gflags lib requires "PathMatchSpecA" defined in "ShLwApi.Lib"
  list(APPEND oneflow_third_party_libs "ShLwApi.Lib")
  list(APPEND oneflow_third_party_libs "Ws2_32.lib")
endif()

set(oneflow_third_party_dependencies
  zlib_copy_headers_to_destination
  zlib_copy_libs_to_destination
  gflags_copy_headers_to_destination
  gflags_copy_libs_to_destination
  glog_copy_headers_to_destination
  glog_copy_libs_to_destination
  googletest_copy_headers_to_destination
  googletest_copy_libs_to_destination
  googlemock_copy_headers_to_destination
  googlemock_copy_libs_to_destination
  protobuf_copy_headers_to_destination
  protobuf_copy_libs_to_destination
  protobuf_copy_binary_to_destination
  grpc_copy_headers_to_destination
  grpc_copy_libs_to_destination
  cub_copy_headers_to_destination
)

include_directories(
    ${ZLIB_INCLUDE_DIR}
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${GOOGLEMOCK_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${GRPC_INCLUDE_DIR}
    ${CUDNN_INCLUDE_DIRS}
    ${CUB_INCLUDE_DIR}
)
