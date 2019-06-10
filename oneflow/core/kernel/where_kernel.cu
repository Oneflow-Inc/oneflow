#include "oneflow/core/kernel/where_kernel.h"
#include <cub/cub.cuh>
#include <math.h>

namespace oneflow {

namespace {

template<typename CondType, typename T>
__global__ void GpuForward(const int64_t elem_cnt, const CondType* condition_ptr, const T* lhs_ptr,
                           const T* rhs_ptr, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out_ptr[i] == static_cast<bool>(condition_ptr[i]) ? rhs_ptr[i] : rhs_ptr[i];
  }
}

template<typename CondType, typename T>
__global__ void GpuBackward(const int64_t elem_cnt, const CondType* condition_ptr,
                            const T* out_diff_ptr, T* lhs_diff_ptr, T* rhs_diff_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    if (static_cast<bool>(condition_ptr[i])) {
      lhs_diff_ptr[i] = out_diff_ptr[i];
      rhs_diff_ptr[i] = 0;
    } else {
      lhs_diff_ptr[i] = 0;
      rhs_diff_ptr[i] = out_diff_ptr[i];
    }
  }
}

}  // namespace

template<typename CondType, typename T>
struct WhereKernelUtil<DeviceType::kGPU, CondType, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const CondType* condition_ptr,
                      const T* lhs_ptr, const T* rhs_ptr, T* out_ptr) {
    GpuForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        elem_cnt, condition_ptr, lhs_ptr, rhs_ptr, out_ptr);
  }

  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const CondType* condition_ptr,
                       const T* out_diff_ptr, T* lhs_diff_ptr, T* rhs_diff_ptr) {
    GpuBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        elem_cnt, condition_ptr, out_diff_ptr, lhs_diff_ptr, rhs_diff_ptr);
  }
};

#define MAKE_ENTRY(cond_type_pair, value_type_pair)                                   \
  template struct WhereKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(cond_type_pair), \
                                  OF_PP_PAIR_FIRST(value_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, INT_DATA_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
