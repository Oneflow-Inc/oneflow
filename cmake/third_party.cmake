if (NOT WIN32)
  find_package(Threads)
endif()

include(zlib)
include(protobuf)
include(googletest)
include(gflags)
include(glog)
include(libjpeg-turbo)
include(opencv)
include(eigen)
include(cocoapi)
include(half)
include(re2)
include(json)
if (RPC_BACKEND STREQUAL "GRPC")
  include(absl)
  include(cares)
  include(openssl)
  include(grpc)
endif()
include(flatbuffers)
include(lz4)

if (WITH_XLA)
  include(tensorflow)
endif()

if (WITH_TENSORRT)
  include(tensorrt)
endif()

if (BUILD_CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
  find_package(CUDA REQUIRED)
  add_definitions(-DWITH_CUDA)
  foreach(cuda_lib_path ${CUDA_LIBRARIES})
    get_filename_component(cuda_lib_name ${cuda_lib_path} NAME)
    if (${cuda_lib_name} STREQUAL libcudart_static.a)
      get_filename_component(cuda_lib_dir ${cuda_lib_path} DIRECTORY)
      break()
    endif()
  endforeach()
  if(NOT EXISTS ${cuda_lib_dir}/libcudart_static.a)
    if(NOT EXISTS ${CUDA_cudart_static_LIBRARY})
      message(FATAL_ERROR "cuda lib not found: ${cuda_lib_dir}/libcudart_static.a")
    endif()
    get_filename_component(cuda_lib_dir ${CUDA_cudart_static_LIBRARY} DIRECTORY)
  endif()
  set(extra_cuda_libs libculibos.a libcurand_static.a)
  if(CUDA_VERSION VERSION_GREATER_EQUAL "10.2")
    list(APPEND extra_cuda_libs libnvjpeg_static.a libnppc_static.a libnppig_static.a)
  endif()
  foreach(extra_cuda_lib ${extra_cuda_libs})
    list(APPEND CUDA_LIBRARIES ${cuda_lib_dir}/${extra_cuda_lib})
  endforeach()
  foreach(cublas_lib_path ${CUDA_CUBLAS_LIBRARIES})
    get_filename_component(cublas_lib_name ${cublas_lib_path} NAME)
    if (${cublas_lib_name} STREQUAL libcublas.so)
      get_filename_component(cublas_lib_dir ${cublas_lib_path} DIRECTORY)
      break()
    endif()
  endforeach()
  if (WITH_XLA)
    if(EXISTS ${cublas_lib_dir}/libcublas.so AND EXISTS ${cublas_lib_dir}/libcublasLt.so)
      list(APPEND CUDA_LIBRARIES ${cublas_lib_dir}/libcublasLt.so)
      list(APPEND CUDA_LIBRARIES ${cublas_lib_dir}/libcublas.so)
    elseif(EXISTS ${cublas_lib_dir}/libcublas.so)
      list(APPEND CUDA_LIBRARIES ${cublas_lib_dir}/libcublas.so)
    elseif(EXISTS ${cuda_lib_dir}/libcublas.so)
      list(APPEND CUDA_LIBRARIES ${cuda_lib_dir}/libcublas.so)
    else()
      message(FATAL_ERROR "cuda lib not found: ${cublas_lib_dir}/libcublas.so or ${cuda_lib_dir}/libcublas.so")
    endif()
  else()
    if(EXISTS ${cublas_lib_dir}/libcublas_static.a AND EXISTS ${cublas_lib_dir}/libcublasLt_static.a)
      list(APPEND CUDA_LIBRARIES ${cublas_lib_dir}/libcublasLt_static.a)
      list(APPEND CUDA_LIBRARIES ${cublas_lib_dir}/libcublas_static.a)
    elseif(EXISTS ${cublas_lib_dir}/libcublas_static.a)
      list(APPEND CUDA_LIBRARIES ${cublas_lib_dir}/libcublas_static.a)
    elseif(EXISTS ${cuda_lib_dir}/libcublas_static.a)
      list(APPEND CUDA_LIBRARIES ${cuda_lib_dir}/libcublas_static.a)
    else()
      message(FATAL_ERROR "cuda lib not found: ${cublas_lib_dir}/libcublas_static.a or ${cuda_lib_dir}/libcublas_static.a")
    endif()
  endif()
  find_package(CUDNN REQUIRED)
endif()

if (NOT WIN32)
  set(BLA_STATIC ON)
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

# libraries only a top level .so or exe should be linked to
set(oneflow_exe_third_party_libs
    ${GLOG_STATIC_LIBRARIES}
    ${GFLAGS_STATIC_LIBRARIES}
)

set(oneflow_third_party_libs
    ${GOOGLETEST_STATIC_LIBRARIES}
    ${GOOGLEMOCK_STATIC_LIBRARIES}
    ${PROTOBUF_STATIC_LIBRARIES}
    ${GRPC_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${OPENCV_STATIC_LIBRARIES}
    ${COCOAPI_STATIC_LIBRARIES}
    ${LIBJPEG_STATIC_LIBRARIES}
    ${ZLIB_STATIC_LIBRARIES}
    ${CARES_STATIC_LIBRARIES}
    ${ABSL_STATIC_LIBRARIES}
    ${OPENSSL_STATIC_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${FLATBUFFERS_STATIC_LIBRARIES}
    ${LZ4_STATIC_LIBRARIES}
)

if (NOT WITH_XLA)
  list(APPEND oneflow_third_party_libs ${RE2_LIBRARIES})
endif()

if(WIN32)
  # static gflags lib requires "PathMatchSpecA" defined in "ShLwApi.Lib"
  list(APPEND oneflow_third_party_libs "ShLwApi.Lib")
  list(APPEND oneflow_third_party_libs "Ws2_32.lib")
endif()

set(oneflow_third_party_dependencies
  zlib_copy_headers_to_destination
  zlib_copy_libs_to_destination
  protobuf_copy_headers_to_destination
  protobuf_copy_libs_to_destination
  protobuf_copy_binary_to_destination
  gflags_copy_headers_to_destination
  gflags_copy_libs_to_destination
  glog_copy_headers_to_destination
  glog_copy_libs_to_destination
  googletest_copy_headers_to_destination
  googletest_copy_libs_to_destination
  googlemock_copy_headers_to_destination
  googlemock_copy_libs_to_destination
  opencv_copy_headers_to_destination
  libpng_copy_headers_to_destination
  opencv_copy_libs_to_destination
  eigen
  cocoapi_copy_headers_to_destination
  cocoapi_copy_libs_to_destination
  half_copy_headers_to_destination
  re2
  json_copy_headers_to_destination
  flatbuffers
  lz4_copy_libs_to_destination
  lz4_copy_headers_to_destination
)


list(APPEND ONEFLOW_INCLUDE_SRC_DIRS
    ${ZLIB_INCLUDE_DIR}
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${GOOGLEMOCK_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${GRPC_INCLUDE_DIR}
    ${LIBJPEG_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${LIBPNG_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${COCOAPI_INCLUDE_DIR}
    ${HALF_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}
    ${CARES_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${LZ4_INCLUDE_DIR}
)

if (NOT WITH_XLA)
  list(APPEND ONEFLOW_INCLUDE_SRC_DIRS ${RE2_INCLUDE_DIR})
endif()

if (RPC_BACKEND STREQUAL "GRPC")
  list(APPEND oneflow_third_party_dependencies grpc_copy_headers_to_destination)
  list(APPEND oneflow_third_party_dependencies grpc_copy_libs_to_destination)
endif()
if (BUILD_CUDA)
  include(cub)
  include(nccl)

  list(APPEND oneflow_third_party_libs ${CUDA_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${CUDNN_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${NCCL_STATIC_LIBRARIES})

  list(APPEND oneflow_third_party_dependencies cub_copy_headers_to_destination)
  list(APPEND oneflow_third_party_dependencies nccl_copy_headers_to_destination)
  list(APPEND oneflow_third_party_dependencies nccl_copy_libs_to_destination)

  list(APPEND ONEFLOW_INCLUDE_SRC_DIRS
    ${CUDNN_INCLUDE_DIRS}
    ${CUB_INCLUDE_DIR}
    ${NCCL_INCLUDE_DIR}
  )
endif()

if(BUILD_RDMA)
  if(UNIX)
    include(CheckIncludeFiles)
    include(CheckLibraryExists)
    CHECK_INCLUDE_FILES(infiniband/verbs.h HAVE_VERBS_H)
    CHECK_LIBRARY_EXISTS(ibverbs ibv_create_qp "" HAVE_IBVERBS)
    if(HAVE_VERBS_H AND HAVE_IBVERBS)
      list(APPEND oneflow_third_party_libs -libverbs)
      add_definitions(-DWITH_RDMA)
    elseif(HAVE_VERBS_H)
      message(FATAL_ERROR "RDMA library not found")
    elseif(HAVE_IBVERBS)
      message(FATAL_ERROR "RDMA head file not found")
    else()
      message(FATAL_ERROR "RDMA library and head file not found")
    endif()
  else()
    message(FATAL_ERROR "UNIMPLEMENTED")
  endif()
endif()

include_directories(${ONEFLOW_INCLUDE_SRC_DIRS})

if(WITH_XLA)
  list(APPEND oneflow_third_party_dependencies tensorflow_copy_libs_to_destination)
  list(APPEND oneflow_third_party_dependencies tensorflow_symlink_headers)
  list(APPEND oneflow_third_party_libs ${TENSORFLOW_XLA_LIBRARIES})
endif()

if(WITH_TENSORRT)
  list(APPEND oneflow_third_party_libs ${TENSORRT_LIBRARIES})
endif()

message(STATUS "oneflow_third_party_libs: " ${oneflow_third_party_libs})

add_definitions(-DHALF_ENABLE_CPP11_USER_LITERALS=0)
