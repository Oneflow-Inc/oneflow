#ifndef ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
struct WhereFunctor {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const;
};

template<DeviceType device_type, typename T, typename CondT>
struct WhereGradFunctor {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* grad,
                  T* lhs_grad, T* rhs_grad) const;
};

template<typename T, typename CondT>
OF_DEVICE_FUNC void DoWhere(const int64_t elem_cnt, const CondT* cond, const T* lhs, const T* rhs,
                            T* out) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? lhs[i] : rhs[i]; }
}

template<typename T, typename CondT>
OF_DEVICE_FUNC void DoWhereGrad(const int64_t elem_cnt, const CondT* cond, const T* grad,
                                T* lhs_grad, T* rhs_grad) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) {
    if (static_cast<bool>(cond[i])) {
      lhs_grad[i] = grad[i];
    } else {
      rhs_grad[i] = grad[i];
    }
  }
}

template<DeviceType device_type, typename T, typename CondT>
class WhereKernel final : public user_op::OpKernel {
 public:
  WhereKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

template<DeviceType device_type, typename T, typename CondT>
class WhereGradKernel final : public user_op::OpKernel {
 public:
  WhereGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  WhereGradKernel() = default;
  ~WhereGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

size_t InferWhereTmpBufferSize(user_op::InferContext* ctx);
size_t InferWhereGradTmpBufferSize(user_op::InferContext* ctx);

#define REGISTER_WHERE_KERNEL(device_type_v, dtype, ctype)                                     \
  REGISTER_USER_KERNEL("where")                                                                \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                                       \
        return new WhereKernel<device_type_v, dtype, ctype>(ctx);                              \
      })                                                                                       \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                             \
        const user_op::TensorDesc* cond_desc = ctx.TensorDesc4ArgNameAndIndex("condition", 0); \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);        \
        return ctx.device_type() == device_type_v                                              \
               && cond_desc->data_type() == GetDataType<ctype>::value                          \
               && out_desc->data_type() == GetDataType<dtype>::value;                          \
      })                                                                                       \
      .SetInferTmpSizeFn(InferWhereTmpBufferSize);

#define REGISTER_WHERE_GRAD_KERNEL(device_type_v, dtype, ctype)                                \
  REGISTER_USER_KERNEL("where_grad")                                                           \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                                       \
        return new WhereGradKernel<device_type_v, dtype, ctype>(ctx);                          \
      })                                                                                       \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                             \
        const user_op::TensorDesc* cond_desc = ctx.TensorDesc4ArgNameAndIndex("condition", 0); \
        const user_op::TensorDesc* dz_desc = ctx.TensorDesc4ArgNameAndIndex("dz", 0);          \
        return ctx.device_type() == device_type_v                                              \
               && cond_desc->data_type() == GetDataType<ctype>::value                          \
               && dz_desc->data_type() == GetDataType<dtype>::value;                           \
      })                                                                                       \
      .SetInferTmpSizeFn(InferWhereGradTmpBufferSize);

#define REGISTER_WHERE_KERNELS(device_type_v, dtype_pair, ctype_pair)                              \
  REGISTER_WHERE_KERNEL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ctype_pair)) \
  REGISTER_WHERE_GRAD_KERNEL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                          \
                             OF_PP_PAIR_FIRST(ctype_pair))

#define INSTANTIATE_WHERE_FUNCTORS(device_type_v, dtype_pair, ctype_pair)       \
  template struct WhereFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),     \
                               OF_PP_PAIR_FIRST(ctype_pair)>;                   \
  template struct WhereGradFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                   OF_PP_PAIR_FIRST(ctype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_H_
