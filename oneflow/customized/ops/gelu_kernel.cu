#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GeluForwardGpu(const int n, const T inv_sqrt2, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5f * x[i] * (1.0f + erff(inv_sqrt2 * x[i])); }
}

template<typename T>
__global__ void GeluBackwardGpu(const int n, const T inv_sqrt2, const T coef, const T* x, const T* dy,
                                     T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = 0.5f * (1.0f + erff(inv_sqrt2 * x[i]) + x[i] * coef * expf(-0.5f * x[i] * x[i])) * dy[i]; }
}

}  // namespace

template<typename T>
class GpuGeluKernel final : public user_op::OpKernel {
 public:
  GpuGeluKernel() = default;
  ~GpuGeluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = in->shape().elem_cnt();
    const T inv_sqrt2 = sqrt(0.5);
    RUN_CUDA_KERNEL((GeluForwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    inv_sqrt2, in->dptr<T>(), out->mut_dptr<T>());
  };
};

#define REGISTER_GPU_GELU_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("gelu")                                                \
      .SetCreateFn<GpuGeluKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_GPU_GELU_KERNEL(float)
REGISTER_GPU_GELU_KERNEL(double)

template<typename T>
class GpuGeluGradKernel final : public user_op::OpKernel {
 public:
  GpuGeluGradKernel() = default;
  ~GpuGeluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T inv_sqrt2 = sqrt(0.5);
    const T coef = sqrt(2.0 / acos(-1.0));
    RUN_CUDA_KERNEL((GeluBackwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, inv_sqrt2,
                    coef, x->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>());
  };
};

#define REGISTER_GPU_GELU_GRAD_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("gelu_grad")                                             \
      .SetCreateFn<GpuGeluGradKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_GELU_GRAD_KERNEL(float)
REGISTER_GPU_GELU_GRAD_KERNEL(double)

}  // namespace oneflow

