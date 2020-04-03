#include "oneflow/customized/kernels/where_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename CondT>
__global__ void CudaWhere(const int64_t elem_cnt, const CondT* cond, const T* lhs, const T* rhs,
                          T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? lhs[i] : rhs[i]; }
}

}  // namespace

template<typename T, typename CondT>
struct WhereKernelUtil<DeviceType::kGPU, T, CondT> {
  static void Where(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                    const T* rhs, T* out) {
    RUN_CUDA_KERNEL((CudaWhere<T, CondT>), ctx, elem_cnt, elem_cnt, cond, lhs, rhs, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
