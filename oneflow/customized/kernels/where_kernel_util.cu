#include "oneflow/customized/kernels/where_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename CondT>
__global__ void CudaWhere(const int64_t elem_cnt, const CondT* cond, const T* lhs, const T* rhs,
                          T* out) {
  DoWhere(elem_cnt, cond, lhs, rhs, out);
}

}  // namespace

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kGPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    RUN_CUDA_KERNEL((CudaWhere<T, CondT>), ctx, elem_cnt, elem_cnt, cond, lhs, rhs, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
