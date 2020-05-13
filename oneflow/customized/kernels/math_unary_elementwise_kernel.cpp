#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/math_unary_elementwise_func.h"

namespace oneflow {

template<template<typename> class UnaryFunctor, typename T>
class MathUnaryElementwiseCpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseCpuKernel() = default;
  ~MathUnaryElementwiseCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x = tensor_x->dptr<T>();
    T* y = tensor_y->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    for (int32_t i = 0; i < n; ++i) { y[i] = UnaryFunctor<T>::Forward(x[i]); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class UnaryFunctor, typename T>
class MathUnaryElementwiseGradCpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseGradCpuKernel() = default;
  ~MathUnaryElementwiseGradCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const T* x = tensor_x->dptr<T>();
    const T* dy = tensor_dy->dptr<T>();
    T* dx = tensor_dx->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    for (int32_t i = 0; i < n; ++i) { dx[i] = UnaryFunctor<T>::Backward(x[i], dy[i]); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD(math_type_pair, data_type_pair)        \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                           \
      .SetCreateFn<                                                                                \
          MathUnaryElementwiseCpuKernel<OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor),     \
                                        OF_PP_PAIR_FIRST(data_type_pair)>>()                       \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                 \
        const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);         \
        const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);         \
        return ctx.device_type() == DeviceType::kCPU                                               \
               && x_tensor_desc->data_type() == OF_PP_PAIR_SECOND(data_type_pair)                  \
               && y_tensor_desc->data_type() == OF_PP_PAIR_SECOND(data_type_pair);                 \
      });                                                                                          \
                                                                                                   \
  REGISTER_USER_KERNEL((std::string("") + OF_PP_PAIR_FIRST(math_type_pair) + "_grad"))             \
      .SetCreateFn<                                                                                \
          MathUnaryElementwiseGradCpuKernel<OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor), \
                                            OF_PP_PAIR_FIRST(data_type_pair)>>()                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                 \
        const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);         \
        return ctx.device_type() == DeviceType::kCPU                                               \
               && x_tensor_desc->data_type() == OF_PP_PAIR_SECOND(data_type_pair);                 \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD,
                                 MATH_UNARY_ELEMENTWISE_FUNC_SEQ, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
