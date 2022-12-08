include(ExternalProject)

find_package(Threads)

set(CUTLASS_PROJECT cutlass)

set(CUTLASS_INSTALL_DIR ${THIRD_PARTY_DIR}/cutlass)

set(CUTLASS_INCLUDE_DIR ${CUTLASS_INSTALL_DIR}/include CACHE PATH "" FORCE)
set(CUTLASS_LIBRARY_DIR ${CUTLASS_INSTALL_DIR}/lib CACHE PATH "" FORCE)
set(CUTLASS_LIBRARIES ${CUTLASS_LIBRARY_DIR}/libcutlass.so)
set(CUTLASS_SOUREC_DIR ${CMAKE_CURRENT_BINARY_DIR}/cutlass/src/cutlass/)

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
               -DBUILD_SHARED_LIBS:BOOL=OFF
               -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
               -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
    CMAKE_CACHE_ARGS
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
      -DCUTLASS_NVCC_EMBED_PTX:BOOL=OFF)

  add_custom_target(cutlass_copy_examples_to_destination DEPENDS cutlass)
  set(CUTLASS_SOURCE_EXAMPLES_DIR ${CUTLASS_SOUREC_DIR}/examples)

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

endif(THIRD_PARTY)
