if(BUILD_CUDA AND USE_HIP)
  # ROCm/HIP backend. BUILD_CUDA stays ON so -DWITH_CUDA is defined
  # and the .cu glob is included; the GPU backend is reached through HIP, with kCUDA
  # still denoting "the GPU backend". The .cu/.cuh translation units compile as HIP. The
  # host .cpp files still compile as C++, but get the same compat header force-included
  # (see set_compile_options_to_oneflow_target), and some carry ROCm-specific changes of
  # their own. See oneflow/core/device/cuda_to_hip.h.
  add_definitions(-DWITH_CUDA)
  add_definitions(-DUSE_HIP)

  # enable_language(HIP) honors an explicit -DCMAKE_HIP_ARCHITECTURES, otherwise
  # auto-detects the host GPU(s), and errors on a no-GPU build host.
  enable_language(HIP)
  message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
  message(STATUS "CMAKE_HIP_COMPILER: ${CMAKE_HIP_COMPILER}")

  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hiprand REQUIRED)
  find_package(rocrand REQUIRED)
  find_package(hipfft REQUIRED)
  find_package(hipcub REQUIRED)
  find_package(rocthrust REQUIRED)
  find_package(rccl REQUIRED) # NCCL -> RCCL; the collective-comm backend links it
  find_package(hipsolver REQUIRED)
  find_package(miopen REQUIRED) # cuDNN -> MIOpen, through the cudnn.h shim header
  find_package(hipblaslt REQUIRED) # cuBLASLt -> hipBLASLt, through the cublasLt.h shim header
  find_package(rocjpeg REQUIRED) # nvJPEG -> rocJPEG (GPU image decoder)
  find_package(rpp REQUIRED) # NPP -> RPP (resize after the GPU image decoder)

  # Link hip::host (the HIP runtime) but NOT hip::device: hip::device carries an
  # INTERFACE_COMPILE_OPTIONS of `-x hip --offload-arch=<arch>` gated on
  # $<COMPILE_LANGUAGE:CXX>, which CMake would then apply to EVERY host .cpp of the
  # monolith -- recompiling them as HIP device code (wrong; they are plain C++). Device
  # codegen for the .cu (LANGUAGE HIP) sources comes from the target's HIP_ARCHITECTURES
  # property, not from hip::device, so dropping hip::device is correct: the host link still
  # gets the runtime via hip::host.
  list(APPEND VENDOR_CUDA_LIBRARIES hip::host)
  list(APPEND VENDOR_CUDA_LIBRARIES roc::hipblas hip::hiprand roc::rocrand hip::hipfft)
  # RCCL satisfies the nccl* symbols the collective-comm GPU backend references (the compat
  # header maps <nccl.h> to <rccl/rccl.h>); roc::rccl pulls only hip::host, no hip::device.
  list(APPEND VENDOR_CUDA_LIBRARIES roc::rccl)
  list(APPEND VENDOR_CUDA_LIBRARIES roc::hipsolver MIOpen roc::hipblaslt)
  list(APPEND VENDOR_CUDA_LIBRARIES rocjpeg::rocjpeg rpp::rpp)
  message(STATUS "VENDOR_CUDA_LIBRARIES (HIP): ${VENDOR_CUDA_LIBRARIES}")

  set(CUDA_VERSION "0.0" CACHE STRING "" FORCE) # unused on HIP; satisfies version guards

  # The ROCm clang finds LLVM's <libunwind.h> ahead of the system one, while FindUnwind
  # finds the system (GNU) libunwind library. The two disagree: GNU's header maps
  # unw_getcontext and friends to _Ux86_64_* macros, LLVM's declares them as plain symbols,
  # so liboneflow.so would reference unw_getcontext, which GNU libunwind does not define,
  # and fail to load. Skip libunwind for both the backtrace code and glog (included later
  # from third_party.cmake). glog then falls back to glibc backtrace() from <execinfo.h>,
  # the backtrace code to _Unwind_Backtrace from <unwind.h>.
  set(CMAKE_DISABLE_FIND_PACKAGE_Unwind ON)

  # Wavefront width for the target arch, visible in BOTH the host and device passes of
  # each .cu (the warp-tiled reductions in core/cuda/{softmax,layer_norm,rms_norm}.cuh
  # use it as a kernel template arg, so it must be one compile-time constant; __GFX9__
  # is set only in the device pass and cannot be used). CDNA (gfx9xx) is 64; RDNA
  # (gfx10xx/gfx11xx) is 32. Other GPUs change only -DCMAKE_HIP_ARCHITECTURES. A build
  # can carry only one width, so an arch list mixing the two families is refused.
  set(_of_wave64_archs)
  set(_of_wave32_archs)
  foreach(_of_arch ${CMAKE_HIP_ARCHITECTURES})
    if("${_of_arch}" MATCHES "^gfx9")
      list(APPEND _of_wave64_archs ${_of_arch})
    else()
      list(APPEND _of_wave32_archs ${_of_arch})
    endif()
  endforeach()
  if(_of_wave64_archs AND _of_wave32_archs)
    message(
      FATAL_ERROR
        "CMAKE_HIP_ARCHITECTURES mixes wavefront-64 (${_of_wave64_archs}) and wavefront-32 "
        "(${_of_wave32_archs}) GPUs. The warp size is a compile-time constant shared by "
        "every code object, so build for one family at a time.")
  endif()
  if(_of_wave64_archs)
    set(_of_warp 64)
  else()
    set(_of_warp 32)
  endif()
  set(OF_GPU_WARP_SIZE ${_of_warp} CACHE INTERNAL "wavefront width for the HIP target arch")
  message(STATUS "OF_GPU_WARP_SIZE: ${OF_GPU_WARP_SIZE}")

  # Force-include the compat header on every HIP TU so its aliases and macro shims
  # precede each file's includes, and put the toolkit-name shim headers on the HIP
  # include path only (the real toolkit headers are absent on ROCm).
  set(OF_HIP_COMPAT_HEADER "${PROJECT_SOURCE_DIR}/oneflow/core/device/cuda_to_hip.h")
  set(OF_HIP_SHIM_DIR "${PROJECT_SOURCE_DIR}/oneflow/core/device/hip_compat")
  string(APPEND CMAKE_HIP_FLAGS " -include ${OF_HIP_COMPAT_HEADER} -I${OF_HIP_SHIM_DIR}")
  string(APPEND CMAKE_HIP_FLAGS " -DOF_GPU_WARP_SIZE=${OF_GPU_WARP_SIZE}")
  string(APPEND CMAKE_HIP_FLAGS " -Wno-unused-result -Wno-deprecated-declarations")

  # The GPU host .cpp TUs (ep/cuda/*.cpp, device/cuda_util.cpp, ...) call cuda*/cublas*
  # and include <cuda.h>/<cuda_runtime.h> too, but are compiled by the CXX compiler, not
  # hipcc. They are given the compat aliases, the shim dir, hip_INCLUDE_DIRS, and the
  # warp-size macro per-target by set_compile_options_to_oneflow_target (NOT globally --
  # a global force-include leaks into the third-party subdir builds, e.g. TBB).
elseif(BUILD_CUDA)
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
  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    set(CUDA_STATIC OFF CACHE BOOL "")
  else()
    set(CUDA_STATIC ON CACHE BOOL "")
  endif()

  if((NOT CUDA_STATIC) OR BUILD_SHARED_LIBS)
    set(OF_CUDA_LINK_DYNAMIC_LIBRARY ON)
  else()
    set(OF_CUDA_LINK_DYNAMIC_LIBRARY OFF)
  endif()

  if(OF_CUDA_LINK_DYNAMIC_LIBRARY)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cublas)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::curand)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cusolver)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cufft)
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
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cufft_static)
    list(APPEND VENDOR_CUDA_LIBRARIES CUDA::cusolver_static)
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
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDNN REQUIRED)

  # NOTE: if you want to use source PTX with a version different from produced PTX/binary, you should add flags
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      # T4, Quadro RTX xxxx, Txxxx, Geforce RTX 20xx, TITAN RTX
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      # A100
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
      # GeForce RTX 30xx
      list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
      # GeForce RTX 40xx
      list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
      # H100, H20
      list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
    endif()
  endif()

  foreach(CUDA_ARCH ${CMAKE_CUDA_ARCHITECTURES})
    if(CUDA_ARCH MATCHES "^([0-9]+)\\-real$")
      list(APPEND CUDA_REAL_ARCHS_LIST ${CMAKE_MATCH_1})
    elseif(CUDA_ARCH MATCHES "^([0-9]+)$")
      list(APPEND CUDA_REAL_ARCHS_LIST ${CMAKE_MATCH_1})
    endif()
  endforeach()

  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  set(CUDA_SEPARABLE_COMPILATION OFF)

  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
      set(CUDA_NVCC_THREADS_NUMBER "4" CACHE STRING "")
      list(APPEND CUDA_NVCC_FLAGS -t ${CUDA_NVCC_THREADS_NUMBER})
    endif()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
    message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
    list(JOIN CUDA_NVCC_FLAGS " " CMAKE_CUDA_FLAGS)
  endif()
endif()
