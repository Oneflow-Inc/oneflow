#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void LeakyReluForwardGpu(const int n, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] >= 0 ? x[i] : x[i] * alpha; }
}

template<typename T>
__global__ void LeakyReluBackwardGpu(const int n, const float alpha, const T* x, const T* dy,
                                     T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = x[i] >= 0 ? dy[i] : dy[i] * alpha; }
}

}  // namespace

template<typename T>
class GpuLeakyReluKernel final : public user_op::OpKernel {
 public:
  GpuLeakyReluKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  GpuLeakyReluKernel() = default;
  ~GpuLeakyReluKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->GetAttr<float>("alpha");
    RUN_CUDA_KERNEL((LeakyReluForwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, alpha,
                    x->dptr<T>(), y->mut_dptr<T>());
  };
};

#define REGISTER_GPU_LEAKY_RELU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("leaky_relu")                                                          \
      .SetCreateFn(                                                                           \
          [](user_op::KernelInitContext* ctx) { return new GpuLeakyReluKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                            \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);           \
        return ctx.device_type() == DeviceType::kGPU                                          \
               && y_desc->data_type() == GetDataType<dtype>::value;                           \
      });

REGISTER_GPU_LEAKY_RELU_KERNEL(float)
REGISTER_GPU_LEAKY_RELU_KERNEL(double)

template<typename T>
class GpuLeakyReluGradKernel final : public user_op::OpKernel {
 public:
  GpuLeakyReluGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  GpuLeakyReluGradKernel() = default;
  ~GpuLeakyReluGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float alpha = ctx->GetAttr<float>("alpha");
    RUN_CUDA_KERNEL((LeakyReluBackwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, alpha,
                    x->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>());
  };
};

#define REGISTER_GPU_LEAKY_RELU_GRAD_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("leaky_relu_grad")                                                         \
      .SetCreateFn(                                                                               \
          [](user_op::KernelInitContext* ctx) { return new GpuLeakyReluGradKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);             \
        return ctx.device_type() == DeviceType::kGPU                                              \
               && dx_desc->data_type() == GetDataType<dtype>::value;                              \
      });

REGISTER_GPU_LEAKY_RELU_GRAD_KERNEL(float)
REGISTER_GPU_LEAKY_RELU_GRAD_KERNEL(double)

}  // namespace oneflow
