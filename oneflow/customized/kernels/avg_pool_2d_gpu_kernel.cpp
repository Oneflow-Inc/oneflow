#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class AvgPool2DGpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2DGpuKernel() = default;
  ~AvgPool2DGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(2, "AVG", ctx);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    // TODO: tsai: reset op kernel state when is_dynamic if ready
    const OpKernelStateWrapper<GPUPoolOpKernelState>* gpu_pool_op_kernel_state =
        dynamic_cast<OpKernelStateWrapper<GPUPoolOpKernelState>*>(state);
    CHECK(gpu_pool_op_kernel_state != nullptr);
    CudaCheck(cudnnPoolingForward(
        ctx->device_ctx()->cudnn_handle(), gpu_pool_op_kernel_state->Get().cudnn_pooling_desc(),
        CudnnSPOnePtr<T>(), gpu_pool_op_kernel_state->Get().cudnn_x_tensor_desc(), x->dptr(),
        CudnnSPZeroPtr<T>(), gpu_pool_op_kernel_state->Get().cudnn_y_tensor_desc(), y->mut_dptr()));
  };
};

#define REGISTER_AVG_POOL_2D_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("avg_pool_2d")                                               \
      .SetCreateFn<AvgPool2DGpuKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_AVG_POOL_2D_GPU_KERNEL(float)
REGISTER_AVG_POOL_2D_GPU_KERNEL(double)

template<typename T>
class AvgPool2DGradGpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2DGradGpuKernel() = default;
  ~AvgPool2DGradGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(2, "AVG", ctx);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    // TODO: tsai: reset op kernel state when is_dynamic if ready
    const OpKernelStateWrapper<GPUPoolOpKernelState>* gpu_pool_op_kernel_state =
        dynamic_cast<OpKernelStateWrapper<GPUPoolOpKernelState>*>(state);
    CHECK(gpu_pool_op_kernel_state != nullptr);
    CudaCheck(cudnnPoolingBackward(
        ctx->device_ctx()->cudnn_handle(), gpu_pool_op_kernel_state->Get().cudnn_pooling_desc(),
        CudnnSPOnePtr<T>(), gpu_pool_op_kernel_state->Get().cudnn_y_tensor_desc(), y->dptr(),
        gpu_pool_op_kernel_state->Get().cudnn_y_tensor_desc(), dy->dptr(),
        gpu_pool_op_kernel_state->Get().cudnn_x_tensor_desc(), x->dptr(), CudnnSPZeroPtr<T>(),
        gpu_pool_op_kernel_state->Get().cudnn_x_tensor_desc(), dx->mut_dptr()));
  };
};

#define REGISTER_AVG_POOL_2D_GRAD_GPU_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                                            \
      .SetCreateFn<AvgPool2DGradGpuKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_AVG_POOL_2D_GRAD_GPU_KERNEL(float)
REGISTER_AVG_POOL_2D_GRAD_GPU_KERNEL(double)

}  // namespace oneflow
