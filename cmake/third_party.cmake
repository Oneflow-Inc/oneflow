cmake_policy(SET CMP0074 NEW)
if (NOT WIN32)
  find_package(Threads REQUIRED)
endif()

# include(zlib)
find_package(ZLIB REQUIRED)
# include(protobuf)
find_package(Protobuf REQUIRED)
# include(googletest)
find_package(GTest REQUIRED)
# include(gflags)
find_package(gflags REQUIRED)
# include(glog)
find_package(glog REQUIRED)
# include(libjpeg-turbo)
# include(opencv)
find_package(OpenCV REQUIRED)
# include(eigen)
find_package(Eigen3 REQUIRED)
if (WITH_COCOAPI)
  include(cocoapi)
endif()
# include(half)
find_package(half REQUIRED)
# include(re2)
find_package(re2 REQUIRED)
# include(json)
find_package(nlohmann_json REQUIRED)
if (RPC_BACKEND MATCHES "GRPC")
  # include(absl)
  # include(cares)
  # include(openssl)
  # include(grpc)
  find_package(gRPC REQUIRED)
endif()
# include(flatbuffers)
find_package(Flatbuffers REQUIRED)
# include(lz4)
find_package(xxHash REQUIRED)

if (WITH_XLA)
  include(tensorflow)
endif()

if (WITH_TENSORRT)
  include(tensorrt)
endif()

# include(hwloc)
find_package(hwloc REQUIRED)

option(CUDA_STATIC "" ON)

if (BUILD_CUDA)
  if ((NOT CUDA_STATIC) OR WITH_XLA OR BUILD_SHARED_LIBS)
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
  if (CUDA_NVCC_GENCODES)
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
    endif()
  endif()
  message(STATUS "VENDOR_CUDA_LIBRARIES: ${VENDOR_CUDA_LIBRARIES}")
  # add a cache entry if want to use a ccache/sccache wrapped nvcc
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE} CACHE STRING "")
  message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
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

set(oneflow_third_party_libs
  gflags::gflags
  glog::glog
  GTest::gtest
  protobuf::libprotobuf
  gRPC::grpc++_unsecure
  opencv::opencv
  ZLIB::ZLIB
  flatbuffers::flatbuffers
  xxHash::xxhash
  nlohmann_json::nlohmann_json
  half::half
  hwloc::hwloc
  Threads::Threads
  ${BLAS_LIBRARIES}
)

if (NOT WITH_XLA)
  list(APPEND oneflow_third_party_libs ${RE2_LIBRARIES})
endif()

if(WIN32)
  # static gflags lib requires "PathMatchSpecA" defined in "ShLwApi.Lib"
  list(APPEND oneflow_third_party_libs "ShLwApi.Lib")
  list(APPEND oneflow_third_party_libs "Ws2_32.lib")
endif()

list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS)

if (NOT WITH_XLA)
  list(APPEND oneflow_third_party_libs re2::re2)
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
  endif()
  include(nccl)

  list(APPEND oneflow_third_party_libs ${VENDOR_CUDA_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${CUDNN_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${NCCL_LIBRARIES})

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
  list(APPEND oneflow_third_party_libs ${HWLOC_STATIC_LIBRARIES})
  list(APPEND oneflow_third_party_libs ${PCIACCESS_STATIC_LIBRARIES})
  list(APPEND ONEFLOW_THIRD_PARTY_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})
  add_definitions(-DWITH_HWLOC)
endif()

include_directories(SYSTEM ${ONEFLOW_THIRD_PARTY_INCLUDE_DIRS})

if(WITH_XLA)
  list(APPEND oneflow_third_party_libs ${TENSORFLOW_XLA_LIBRARIES})
endif()

if(WITH_TENSORRT)
  list(APPEND oneflow_third_party_libs ${TENSORRT_LIBRARIES})
endif()

message(STATUS "oneflow_third_party_libs: ${oneflow_third_party_libs}")

add_definitions(-DHALF_ENABLE_CPP11_USER_LITERALS=0)
