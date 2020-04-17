#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class MaxPool3DGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DGpuKernel() = default;
  ~MaxPool3DGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(2, "MAX", ctx);
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

#define REGISTER_MAX_POOL_3D_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("max_pool_3d")                                               \
      .SetCreateFn<MaxPool3DGpuKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_MAX_POOL_3D_GPU_KERNEL(float)
REGISTER_MAX_POOL_3D_GPU_KERNEL(double)

template<typename T>
class MaxPool3DGradGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DGradGpuKernel() = default;
  ~MaxPool3DGradGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(2, "MAX", ctx);
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

#define REGISTER_MAX_POOL_3D_GRAD_GPU_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                                            \
      .SetCreateFn<MaxPool3DGradGpuKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_MAX_POOL_3D_GRAD_GPU_KERNEL(float)
REGISTER_MAX_POOL_3D_GRAD_GPU_KERNEL(double)

}  // namespace oneflow
