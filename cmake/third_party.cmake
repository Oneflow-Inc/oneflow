cmake_policy(SET CMP0074 NEW)
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
if (WITH_COCOAPI)
  include(cocoapi)
endif()
include(half)
include(re2)
include(json)
if (RPC_BACKEND MATCHES "GRPC")
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

include(hwloc)

option(CUDA_STATIC "" ON)

if (BUILD_CUDA)
  if ((NOT CUDA_STATIC) OR WITH_XLA OR BUILD_SHARED_LIBS)
    set(OF_CUDA_LINK_DYNAMIC_LIBRARY ON)
  else()
    set(OF_CUDA_LINK_DYNAMIC_LIBRARY OFF)
  endif()
  if(OF_CUDA_LINK_DYNAMIC_LIBRARY)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  endif()
  find_package(CUDA REQUIRED)
  add_definitions(-DWITH_CUDA)
  set(VENDOR_CUDA_LIBRARIES ${CUDA_LIBRARIES})
  if(OF_CUDA_LINK_DYNAMIC_LIBRARY)
    list(APPEND VENDOR_CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
    list(APPEND VENDOR_CUDA_LIBRARIES ${CUDA_curand_LIBRARY})
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.2")
      find_cuda_helper_libs(nvjpeg)
      list(APPEND VENDOR_CUDA_LIBRARIES ${CUDA_nvjpeg_LIBRARY})
      list(APPEND VENDOR_CUDA_LIBRARIES ${CUDA_nppc_LIBRARY})
      list(APPEND VENDOR_CUDA_LIBRARIES ${CUDA_nppig_LIBRARY})
    endif()
  else()
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
      list(APPEND VENDOR_CUDA_LIBRARIES ${cuda_lib_dir}/${extra_cuda_lib})
    endforeach()
    foreach(cublas_lib_path ${CUDA_CUBLAS_LIBRARIES})
      get_filename_component(cublas_lib_name ${cublas_lib_path} NAME)
      if (${cublas_lib_name} STREQUAL libcublas.so)
        get_filename_component(cublas_lib_dir ${cublas_lib_path} DIRECTORY)
        break()
      endif()
    endforeach()
    if(EXISTS ${cublas_lib_dir}/libcublas_static.a AND EXISTS ${cublas_lib_dir}/libcublasLt_static.a)
      list(APPEND VENDOR_CUDA_LIBRARIES ${cublas_lib_dir}/libcublasLt_static.a)
      list(APPEND VENDOR_CUDA_LIBRARIES ${cublas_lib_dir}/libcublas_static.a)
    elseif(EXISTS ${cublas_lib_dir}/libcublas_static.a)
      list(APPEND VENDOR_CUDA_LIBRARIES ${cublas_lib_dir}/libcublas_static.a)
    elseif(EXISTS ${cuda_lib_dir}/libcublas_static.a)
      list(APPEND VENDOR_CUDA_LIBRARIES ${cuda_lib_dir}/libcublas_static.a)
    else()
      message(FATAL_ERROR "cuda lib not found: ${cublas_lib_dir}/libcublas_static.a or ${cuda_lib_dir}/libcublas_static.a")
    endif()
  endif()
  set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
  set(CMAKE_CUDA_STANDARD 11)
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
    glog_imported
    gflags_imported
)

set(oneflow_third_party_libs
    ${GOOGLETEST_STATIC_LIBRARIES}
    ${GOOGLEMOCK_STATIC_LIBRARIES}
    protobuf_imported
    ${GRPC_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${OPENCV_STATIC_LIBRARIES}
    ${COCOAPI_STATIC_LIBRARIES}
    ${LIBJPEG_STATIC_LIBRARIES}
    zlib_imported
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
  zlib
  protobuf
  gflags
  glog
  googletest_copy_headers_to_destination
  googletest_copy_libs_to_destination
  googlemock_copy_headers_to_destination
  googlemock_copy_libs_to_destination
  opencv_copy_headers_to_destination
  libpng_copy_headers_to_destination
  opencv_copy_libs_to_destination
  eigen
  half_copy_headers_to_destination
  re2
  json_copy_headers_to_destination
  flatbuffers
  lz4_copy_libs_to_destination
  lz4_copy_headers_to_destination
)

if (WITH_COCOAPI)
  list(APPEND oneflow_third_party_dependencies cocoapi_copy_headers_to_destination)
  list(APPEND oneflow_third_party_dependencies cocoapi_copy_libs_to_destination)
endif()

if (RPC_BACKEND MATCHES "GRPC")
  list(APPEND oneflow_third_party_dependencies grpc)
endif()

list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS
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
    ${OPENSSL_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${LZ4_INCLUDE_DIR}
)

if (NOT WITH_XLA)
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${RE2_INCLUDE_DIR})
endif()

if (BUILD_CUDA)
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

  list(APPEND oneflow_third_party_libs ${VENDOR_CUDA_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${CUDNN_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${NCCL_LIBRARIES})

  list(APPEND oneflow_third_party_dependencies nccl)

  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS
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
  list(APPEND oneflow_third_party_libs ${HWLOC_STATIC_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${PCIACCESS_STATIC_LIBRARIES})
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})
  add_definitions(-DWITH_HWLOC)
endif()

include_directories(SYSTEM ${ONEFLOW_THIRD_PARTY_INCLUDE_DIRS})

if(WITH_XLA)
  list(APPEND oneflow_third_party_dependencies tensorflow_copy_libs_to_destination)
  list(APPEND oneflow_third_party_dependencies tensorflow_symlink_headers)
  list(APPEND oneflow_third_party_libs ${TENSORFLOW_XLA_LIBRARIES})
endif()

if(WITH_TENSORRT)
  list(APPEND oneflow_third_party_libs ${TENSORRT_LIBRARIES})
endif()

message(STATUS "oneflow_third_party_libs: ${oneflow_third_party_libs}")

add_definitions(-DHALF_ENABLE_CPP11_USER_LITERALS=0)

if (THIRD_PARTY)
  add_custom_target(prepare_oneflow_third_party ALL DEPENDS ${oneflow_third_party_dependencies})
  foreach(of_include_src_dir ${ONEFLOW_THIRD_PARTY_INCLUDE_DIRS})
    set(oneflow_all_include_file)
    file(GLOB_RECURSE oneflow_all_include_file "${of_include_src_dir}/*.*")
    copy_files("${oneflow_all_include_file}" "${of_include_src_dir}" "${ONEFLOW_INCLUDE_DIR}" prepare_oneflow_third_party)
  endforeach()
else()
  add_custom_target(prepare_oneflow_third_party ALL)
endif()
