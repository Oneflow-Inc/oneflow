if (NOT WIN32)
  find_package(Threads)
endif()

include(zlib)
include(protobuf)
include(googletest)
include(gflags)
include(glog)
include(grpc)
include(libjpeg-turbo)
include(opencv)
include(eigen)
include(cocoapi)
include(half)
include(json)

if (WITH_XLA)
  include(tensorflow)
endif()

if (WITH_TENSORRT)
  if (NOT WITH_XLA)
    include(absl)
  endif()
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
  set(extra_cuda_libs libculibos.a libcublas_static.a libcurand_static.a)
  foreach(extra_cuda_lib ${extra_cuda_libs})
    list(APPEND CUDA_LIBRARIES ${cuda_lib_dir}/${extra_cuda_lib})
  endforeach()
  find_package(CuDNN REQUIRED)
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
    ${BLAS_LIBRARIES}
    ${LIBJPEG_STATIC_LIBRARIES}
    ${OPENCV_STATIC_LIBRARIES}
    ${COCOAPI_STATIC_LIBRARIES}
)

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
  grpc_copy_headers_to_destination
  grpc_copy_libs_to_destination
  opencv_copy_headers_to_destination
  opencv_copy_libs_to_destination
  eigen
  cocoapi_copy_headers_to_destination
  cocoapi_copy_libs_to_destination
  half_copy_headers_to_destination
  json_copy_headers_to_destination
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
    ${EIGEN_INCLUDE_DIR}
    ${COCOAPI_INCLUDE_DIR}
    ${HALF_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

if (BUILD_CUDA)
  include(cub)
  include(nccl)

  if (WITH_XLA)
    # Fix conflicts between tensorflow cublas dso and oneflow static cublas.
    # TODO(hjchen2) Should commit a issue about this fix.
    list(APPEND oneflow_third_party_libs -Wl,--whole-archive ${cuda_lib_dir}/libcublas_static.a -Wl,--no-whole-archive)
  endif()
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
  list(APPEND oneflow_third_party_libs ${TENSORFLOW_XLA_LIBRARIES})
endif()

if(WITH_TENSORRT)
  if (NOT WITH_XLA)
    list(APPEND oneflow_third_party_libs ${ABSL_LIBRARIES})
  endif()
  list(APPEND oneflow_third_party_libs ${TENSORRT_LIBRARIES})
endif()

message(STATUS "oneflow_third_party_libs: " ${oneflow_third_party_libs})

add_definitions(-DHALF_ENABLE_CPP11_USER_LITERALS=0)
