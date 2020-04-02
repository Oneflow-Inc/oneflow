#include "oneflow/customized/kernels/where_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
void WhereKernel<device_type, T, CondT>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  WhereFunctor<device_type, T, CondT>()(ctx->device_ctx(), out->shape().elem_cnt(),
                                        cond->dptr<CondT>(), x->dptr<T>(), y->dptr<T>(),
                                        out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename CondT>
void WhereGradKernel<device_type, T, CondT>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
  const user_op::Tensor* dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
  user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
  user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
  size_t dx_bytes = GetCudaAlignedSize(dx->shape().elem_cnt() * sizeof(T));
  size_t dy_bytes = GetCudaAlignedSize(dy->shape().elem_cnt() * sizeof(T));
  CHECK_EQ(dx_bytes, dy_bytes);
  Memset<device_type>(ctx->device_ctx(), dx->mut_dptr<T>(), 0, dx_bytes);
  Memset<device_type>(ctx->device_ctx(), dy->mut_dptr<T>(), 0, dy_bytes);
  WhereGradFunctor<device_type, T, CondT>()(ctx->device_ctx(), dz->shape().elem_cnt(),
                                            cond->dptr<CondT>(), dz->dptr<T>(), dx->mut_dptr<T>(),
                                            dy->mut_dptr<T>());
}

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kCPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    DoWhere(elem_cnt, cond, lhs, rhs, out);
  }
};

template<typename T, typename CondT>
struct WhereGradFunctor<DeviceType::kCPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* grad,
                  T* lhs_grad, T* rhs_grad) const {
    DoWhereGrad(elem_cnt, cond, grad, lhs_grad, rhs_grad);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTORS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_KERNELS, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)

}  // namespace oneflow
