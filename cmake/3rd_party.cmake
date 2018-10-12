if (NOT WIN32)
  find_package(Threads)
endif()

set(THIRD_PARTY_URL http://download.oneflow.org/third_party_without_cuda.tgz)
include(zlib)
include(protobuf)

if (NOT WIN32)
  message(STATUS "downloading third_partys ...")
  file(DOWNLOAD ${THIRD_PARTY_URL} ${PROJECT_BINARY_DIR}/third_party.tgz SHOW_PROGRESS)
else()
  message(STATUS "win32 not support yet.")
endif()

if (BUILD_CUDA)
  # download without_cub_third_party
  # TODO:
  include(cub)
  set(THIRD_PARTY_URL http://download.oneflow.org/third_party_without_cuda.tgz)
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
  if (BUILD_NCCL)
    find_package(NCCL REQUIRED)
    if (NCCL_VERSION VERSION_LESS 2.0)
      message(FATAL_ERROR "minimum nccl version required is 2.0")
    else()
      add_definitions(-DWITH_NCCL)
    endif()
  endif()
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

if (BUILD_CUDA)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar zxvf ${PROJECT_BINARY_DIR}/third_party.tgz
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endif()

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
    ${CUDNN_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LIBJPEG_STATIC_LIBRARIES}
    ${OPENCV_STATIC_LIBRARIES}
    ${CMAKE_DL_LIBS}
)
message(STATUS "oneflow_third_party_libs: " ${oneflow_third_party_libs})

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
  opencv_copy_headers_to_destination
  opencv_copy_libs_to_destination
  eigen
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
    ${LIBJPEG_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)
