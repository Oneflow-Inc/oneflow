include(ExternalProject)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(WITH_CUTLASS_INIT OFF)
else()
  set(WITH_CUTLASS_INIT ON)
endif()

set(WITH_CUTLASS ${WITH_CUTLASS_INIT} CACHE BOOL "")
set(WITH_OF_FLASH_ATTENTION ON CACHE BOOL "")

if(WITH_CUTLASS)

  add_definitions(-DWITH_CUTLASS)

  find_package(Threads)

  set(CUTLASS_PROJECT cutlass)

  set(CUTLASS_INSTALL_DIR ${THIRD_PARTY_DIR}/cutlass)

  set(CUTLASS_INCLUDE_DIR ${CUTLASS_INSTALL_DIR}/include CACHE PATH "" FORCE)
  set(CUTLASS_LIBRARY_DIR ${CUTLASS_INSTALL_DIR}/lib CACHE PATH "" FORCE)
  set(CUTLASS_LIBRARIES ${CUTLASS_LIBRARY_DIR}/libcutlass.so)
  set(CUTLASS_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cutlass/src/cutlass/)

  foreach(arch ${CUDA_REAL_ARCHS_LIST})
    if(arch GREATER_EQUAL 70)
      list(APPEND CUTLASS_REAL_ARCHS ${arch})
    endif()
  endforeach()

  if(THIRD_PARTY)
    ExternalProject_Add(
      ${CUTLASS_PROJECT}
      PREFIX cutlass
      URL ${CUTLASS_URL}
      URL_MD5 ${CUTLASS_MD5}
      UPDATE_COMMAND ""
      BUILD_BYPRODUCTS ${CUTLASS_LIBRARIES}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                 -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
                 -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
                 -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
      CMAKE_CACHE_ARGS
        -DCMAKE_CUDA_COMPILER:STRING=${CUDAToolkit_NVCC_EXECUTABLE}
        -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_INSTALL_PREFIX:PATH=${CUTLASS_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR:PATH=${CUTLASS_LIBRARY_DIR}
        -DCMAKE_INSTALL_MESSAGE:STRING=${CMAKE_INSTALL_MESSAGE}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCUTLASS_LIBRARY_OPERATIONS:STRING=conv2d
        -DCUTLASS_LIBRARY_KERNELS:STRING=simt_hfprop_*,tensorop_f16_*fprop,tensorop_h*fprop
        -DCUTLASS_ENABLE_EXAMPLES:BOOL=OFF
        -DCUTLASS_ENABLE_PROFILER:BOOL=OFF
        -DCUTLASS_ENABLE_LIBRARY:BOOL=ON
        -DCUTLASS_NVCC_ARCHS:STRING=${CUTLASS_REAL_ARCHS}
        -DCUTLASS_ENABLE_TESTS:BOOL=OFF
        -DCUTLASS_UNITY_BUILD_ENABLED:BOOL=ON
        -DCUTLASS_LIBRARY_DEBUG_POSTFIX:STRING=
        -DCUTLASS_NVCC_EMBED_PTX:BOOL=OFF)

    add_custom_target(cutlass_copy_examples_to_destination DEPENDS cutlass)
    set(CUTLASS_SOURCE_EXAMPLES_DIR ${CUTLASS_SOURCE_DIR}/examples)

    set(CUTLASS_INSTALL_EXAMPLES_FILES
        "41_fused_multi_head_attention/iterators/make_residual_last.h"
        "41_fused_multi_head_attention/iterators/epilogue_predicated_tile_iterator.h"
        "41_fused_multi_head_attention/iterators/predicated_tile_iterator_residual_last.h"
        "41_fused_multi_head_attention/iterators/predicated_tile_access_iterator_residual_last.h"
        "41_fused_multi_head_attention/mma_from_smem.h"
        "41_fused_multi_head_attention/epilogue_rescale_output.h"
        "41_fused_multi_head_attention/attention_scaling_coefs_updater.h"
        "41_fused_multi_head_attention/gemm_kernel_utils.h"
        "41_fused_multi_head_attention/fmha_grouped_problem_visitor.h"
        "41_fused_multi_head_attention/fmha_grouped.h"
        "41_fused_multi_head_attention/default_fmha_grouped.h"
        "41_fused_multi_head_attention/epilogue_pipelined.h"
        "41_fused_multi_head_attention/epilogue_thread_apply_logsumexp.h"
        "41_fused_multi_head_attention/kernel_forward.h"
        "41_fused_multi_head_attention/gemm/custom_mma_multistage.h"
        "41_fused_multi_head_attention/gemm/custom_mma_base.h"
        "41_fused_multi_head_attention/gemm/custom_mma.h"
        "41_fused_multi_head_attention/gemm/custom_mma_pipelined.h"
        "41_fused_multi_head_attention/find_default_mma.h"
        "41_fused_multi_head_attention/debug_utils.h"
        "45_dual_gemm/test_run.h"
        "45_dual_gemm/kernel/dual_gemm.h"
        "45_dual_gemm/device/dual_gemm.h"
        "45_dual_gemm/dual_gemm_run.h"
        "45_dual_gemm/thread/left_silu_and_mul.h"
        "45_dual_gemm/threadblock/dual_mma_multistage.h"
        "45_dual_gemm/threadblock/dual_epilogue.h"
        "45_dual_gemm/threadblock/dual_mma_base.h")

    foreach(filename ${CUTLASS_INSTALL_EXAMPLES_FILES})
      add_custom_command(
        TARGET cutlass_copy_examples_to_destination
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUTLASS_SOURCE_EXAMPLES_DIR}/${filename}
                ${CUTLASS_INSTALL_DIR}/examples/${filename})
    endforeach()

    if(WITH_OF_FLASH_ATTENTION)
      set(OF_FLASH_ATTENTION_INSTALL_DIR ${THIRD_PARTY_DIR}/flash_attention)
      set(OF_FLASH_ATTENTION_INCLUDE_DIR ${OF_FLASH_ATTENTION_INSTALL_DIR}/include/csrc/flash_attn/src)
      FetchContent_Declare(
          flash-attention
          URL     https://github.com/Oneflow-Inc/flash-attention/archive/58e98e3492c14fa9f2fd1de5fe9056a14dc6400c.zip
          URL_HASH MD5=7c3760af96534b68a243d6efda689510
          SOURCE_DIR ${OF_FLASH_ATTENTION_INSTALL_DIR}/include
      )
      FetchContent_MakeAvailable(flash-attention)
      set(OF_FLASH_ATTENTION_SRC_FILES 
          ${OF_FLASH_ATTENTION_INCLUDE_DIR}/fmha_fwd_hdim32.cu
          ${OF_FLASH_ATTENTION_INCLUDE_DIR}/fmha_fwd_hdim64.cu
          ${OF_FLASH_ATTENTION_INCLUDE_DIR}/fmha_fwd_hdim128.cu
          ${OF_FLASH_ATTENTION_INCLUDE_DIR}/fmha_bwd_hdim32.cu
          ${OF_FLASH_ATTENTION_INCLUDE_DIR}/fmha_bwd_hdim64.cu
          ${OF_FLASH_ATTENTION_INCLUDE_DIR}/fmha_bwd_hdim128.cu)

      add_library(of_flash_attention ${OF_FLASH_ATTENTION_SRC_FILES})
      add_dependencies(of_flash_attention cutlass)
      target_compile_options(of_flash_attention PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                              --expt-relaxed-constexpr --use_fast_math
                          >)
      set(OF_FLASH_ATTENTION_LIBRARIES ${PROJECT_BINARY_DIR}/libof_flash_attention.a)
      set_target_properties(of_flash_attention PROPERTIES CUDA_ARCHITECTURES "75;80;86")
      target_include_directories(of_flash_attention PUBLIC $<BUILD_INTERFACE:${OF_FLASH_ATTENTION_INCLUDE_DIR}>)
      target_include_directories(of_flash_attention PRIVATE ${CUTLASS_INCLUDE_DIR})
    endif(WITH_OF_FLASH_ATTENTION)
  endif(THIRD_PARTY)
endif(WITH_CUTLASS)
