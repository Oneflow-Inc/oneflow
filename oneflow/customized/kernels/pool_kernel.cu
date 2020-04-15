#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

namespace {

std::shared_ptr<user_op::OpKernelState> DoCreateOpKernelState(user_op::KernelInitContext* ctx) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const int32_t dim = ctx->GetAttr<int32_t>("dim");
  const std::string pooling_type = ctx->GetAttr<std::string>("pooling_type");
  const std::string data_format = ctx->GetAttr<std::string>("data_format");
  const std::string padding = ctx->GetAttr<std::string>("padding");
  const std::vector<int32_t>& pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
  const std::vector<int32_t>& strides = ctx->GetAttr<std::vector<int32_t>>("strides");
  const Params3D params_3d(dim, x_shape, data_format, padding, pool_size, strides);
  const Shape y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();
  const DataType dtype = ctx->TensorDesc4ArgNameAndIndex("x", 0)->data_type();
  return std::make_shared<OpKernelStateWrapper<GPUPoolOpKernelState>>(
      dim, pooling_type, x_shape, y_shape, data_format, dtype, params_3d);
}

}  // namespace

template<typename T>
class GPUPoolKernel final : public user_op::OpKernel {
 public:
  GPUPoolKernel() = default;
  ~GPUPoolKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx);
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

#define REGISTER_GPU_POOL_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("pool").SetCreateFn<GPUPoolKernel<dtype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                     \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);  \
        return ctx.device_type() == DeviceType::kGPU                                 \
               && y_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_POOL_KERNEL(float)
REGISTER_GPU_POOL_KERNEL(double)

template<typename T>
class GpuPoolGradKernel final : public user_op::OpKernel {
 public:
  GpuPoolGradKernel() = default;
  ~GpuPoolGradKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx);
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
        x.Get(), dx->mut_dptr()));
  };
};

#define REGISTER_GPU_POOL_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("pool_grad")                                                   \
      .SetCreateFn<GpuPoolGradKernel<dtype>>()                                        \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_POOL_KERNEL(float)
REGISTER_GPU_POOL_KERNEL(double)

}  // namespace oneflow
