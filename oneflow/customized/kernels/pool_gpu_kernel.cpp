#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

namespace {

template<typename dtype>
std::function<bool(const user_op::KernelRegContext& ctx)> MakeIsMatchedPred(
    DeviceType device_type) {
  return [device_type](const user_op::KernelRegContext& ctx) {
    const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
    return ctx.device_type() == device_type && y_desc->data_type() == GetDataType<dtype>::value;
  };
}
}  // namespace

template<typename T>
class AvgPool1DGpuKernel final : public user_op::OpKernel {
 public:
  AvgPool1DGpuKernel() = default;
  ~AvgPool1DGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(1, "AVG", ctx);
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

template<typename T>
class AvgPool1DGradGpuKernel final : public user_op::OpKernel {
 public:
  AvgPool1DGradGpuKernel() = default;
  ~AvgPool1DGradGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(1, "AVG", ctx);
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

template<typename T>
class AvgPool3DGpuKernel final : public user_op::OpKernel {
 public:
  AvgPool3DGpuKernel() = default;
  ~AvgPool3DGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(3, "AVG", ctx);
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

template<typename T>
class AvgPool3DGradGpuKernel final : public user_op::OpKernel {
 public:
  AvgPool3DGradGpuKernel() = default;
  ~AvgPool3DGradGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(3, "AVG", ctx);
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

template<typename T>
class MaxPool1DGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool1DGpuKernel() = default;
  ~MaxPool1DGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(1, "MAX", ctx);
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

template<typename T>
class MaxPool1DGradGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool1DGradGpuKernel() = default;
  ~MaxPool1DGradGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(1, "MAX", ctx);
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

template<typename T>
class MaxPool2DGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2DGpuKernel() = default;
  ~MaxPool2DGpuKernel() = default;

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

template<typename T>
class MaxPool2DGradGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2DGradGpuKernel() = default;
  ~MaxPool2DGradGpuKernel() = default;

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

template<typename T>
class MaxPool3DGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DGpuKernel() = default;
  ~MaxPool3DGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(3, "MAX", ctx);
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

template<typename T>
class MaxPool3DGradGpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DGradGpuKernel() = default;
  ~MaxPool3DGradGpuKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return GPUPoolOpKernelState::FromKernelInitContext(3, "MAX", ctx);
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

#define REGISTER_POOL_GPU_KERNEL(dtype)                              \
  REGISTER_USER_KERNEL("avg_pool_1d")                                \
      .SetCreateFn<AvgPool1DGpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("avg_pool_1d_grad")                           \
      .SetCreateFn<AvgPool1DGradGpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("avg_pool_2d")                                \
      .SetCreateFn<AvgPool2DGpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                           \
      .SetCreateFn<AvgPool2DGradGpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("avg_pool_3d")                                \
      .SetCreateFn<AvgPool3DGpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("avg_pool_3d_grad")                           \
      .SetCreateFn<AvgPool3DGradGpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("max_pool_1d")                                \
      .SetCreateFn<MaxPool1DGpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("max_pool_1d_grad")                           \
      .SetCreateFn<MaxPool1DGradGpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("max_pool_2d")                                \
      .SetCreateFn<MaxPool2DGpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("max_pool_2d_grad")                           \
      .SetCreateFn<MaxPool2DGradGpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("max_pool_3d")                                \
      .SetCreateFn<MaxPool3DGpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU)); \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                           \
      .SetCreateFn<MaxPool3DGradGpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kGPU));

REGISTER_POOL_GPU_KERNEL(float)
REGISTER_POOL_GPU_KERNEL(double)

}  // namespace oneflow
