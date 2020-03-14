#include "oneflow/customized/kernels/where_kernel.h"

namespace oneflow {

namespace {

template<typename T, typename CondT>
__global__ void CudaWhere(const int64_t elem_cnt, const CondT* cond, const T* lhs, const T* rhs,
                          T* out) {
  DoWhere(elem_cnt, cond, lhs, rhs, out);
}

template<typename T, typename CondT>
__global__ void CudaWhereGrad(const int64_t elem_cnt, const CondT* cond, const T* grad, T* lhs_grad,
                              T* rhs_grad) {
  DoWhere(elem_cnt, cond, grad, lhs_grad, rhs_grad);
}

}  // namespace

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kGPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    CudaWhere<T, CondT>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, cond, lhs, rhs, out);
  }
};

template<typename T, typename CondT>
struct WhereGradFunctor<DeviceType::kGPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* grad,
                  T* lhs_grad, T* rhs_grad) const {
    CudaWhereGrad<T, CondT>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, cond, grad, lhs_grad, rhs_grad);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTORS, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
