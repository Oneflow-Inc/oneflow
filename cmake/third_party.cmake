if (NOT WIN32)
  find_package(Threads)
endif()

include(zlib)
include(protobuf)
include(googletest)
include(glog)
include(gflags)
include(grpc)
include(tensorflow)
include(cub)

find_package(CUDA REQUIRED)
find_package(CuDNN REQUIRED)

if (NOT WIN32)
  set(BLA_VENDOR "Intel10_64lp_seq")
  find_package(BLAS REQUIRED)
else()
  set(MKL_LIB_PATH "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2017/windows/mkl/lib/intel64_win")
  set(BLAS_LIBRARIES ${MKL_LIB_PATH}/mkl_core_dll.lib ${MKL_LIB_PATH}/mkl_sequential_dll.lib ${MKL_LIB_PATH}/mkl_intel_lp64_dll.lib)
endif()
message(STATUS "Blas Lib: " ${BLAS_LIBRARIES})

set(oneflow_third_party_libs
    ${tensorflow_STATIC_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${ZLIB_STATIC_LIBRARIES}
    ${GLOG_STATIC_LIBRARIES}
    ${GFLAGS_STATIC_LIBRARIES}
    ${GOOGLETEST_STATIC_LIBRARIES}
    ${GOOGLEMOCK_STATIC_LIBRARIES}
    ${PROTOBUF_STATIC_LIBRARIES}
    ${GRPC_STATIC_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${highwayhash_STATIC_LIBRARIES}
    ${JPEG_STATIC_LIBRARIES}
    ${PNG_STATIC_LIBRARIES}
    ${JSONCPP_STATIC_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${BLAS_LIBRARIES}
)

if(WIN32)
  # static gflags lib requires "PathMatchSpecA" defined in "ShLwApi.Lib"
  list(APPEND oneflow_third_party_libs "ShLwApi.Lib")
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
  tensorflow_copy_headers_to_destination
  tensorflow_copy_libs_to_destination
  gif_copy_headers_to_destination
  gif_copy_libs_to_destination
  farmhash_copy_headers_to_destination
  farmhash_copy_libs_to_destination
  highwayhash_copy_headers_to_destination
  highwayhash_copy_libs_to_destination
  jpeg_copy_headers_to_destination
  jpeg_copy_libs_to_destination
  png_copy_headers_to_destination
  png_copy_libs_to_destination
  jsoncpp_copy_headers_to_destination
  jsoncpp_copy_libs_to_destination
  eigen_copy_headers_dir
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
    ${TENSORFLOW_INCLUDE_DIR}
    ${GIF_INCLUDE_DIR}
    ${FARMHASH_INCLUDE_DIR}
    ${HIGHWAYHASH_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${PNG_INCLUDE_DIR}
    ${JSONCPP_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}
    ${CUB_INCLUDE_DIR}
)
