#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class CpuLeakyReluKernel final : public user_op::OpKernel {
 public:
  CpuLeakyReluKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  CpuLeakyReluKernel() = default;
  ~CpuLeakyReluKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->GetAttr<float>("alpha");
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = x_ptr[i] >= 0 ? x_ptr[i] : x_ptr[i] * alpha; }
  };
};

#define REGISTER_CPU_LEAKY_RELU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("leaky_relu")                                                          \
      .SetCreateFn(                                                                           \
          [](user_op::KernelInitContext* ctx) { return new CpuLeakyReluKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                            \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);           \
        return ctx.device_type() == DeviceType::kCPU                                          \
               && y_desc->data_type() == GetDataType<dtype>::value;                           \
      });

REGISTER_CPU_LEAKY_RELU_KERNEL(float)
REGISTER_CPU_LEAKY_RELU_KERNEL(double)

template<typename T>
class CpuLeakyReluGradKernel final : public user_op::OpKernel {
 public:
  CpuLeakyReluGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  CpuLeakyReluGradKernel() = default;
  ~CpuLeakyReluGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->GetAttr<float>("alpha");
    const T* x_ptr = x->dptr<T>();
    const T* dy_ptr = dy->dptr<T>();
    T* dx_ptr = dx->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] = x_ptr[i] >= 0 ? dy_ptr[i] : dy_ptr[i] * alpha;
    }
  };
};

#define REGISTER_CPU_LEAKY_RELU_GRAD_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("leaky_relu_grad")                                                         \
      .SetCreateFn(                                                                               \
          [](user_op::KernelInitContext* ctx) { return new CpuLeakyReluGradKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);             \
        return ctx.device_type() == DeviceType::kCPU                                              \
               && dx_desc->data_type() == GetDataType<dtype>::value;                              \
      });

REGISTER_CPU_LEAKY_RELU_GRAD_KERNEL(float)
REGISTER_CPU_LEAKY_RELU_GRAD_KERNEL(double)

}  // namespace oneflow
