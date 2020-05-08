#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {

class CudnnPoolDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolDesc);
  CudnnPoolDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window, const int* padding,
                const int* stride) {
    CudaCheck(cudnnCreatePoolingDescriptor(&val_));
    CudaCheck(cudnnSetPoolingNdDescriptor(val_, pooling_mode, CUDNN_NOT_PROPAGATE_NAN, dims, window,
                                          padding, stride));
  }

  ~CudnnPoolDesc() { CudaCheck(cudnnDestroyPoolingDescriptor(val_)); }

  const cudnnPoolingDescriptor_t& Get() const { return val_; }

 private:
  cudnnPoolingDescriptor_t val_;
};

class GPUPoolOpKernelState final : public user_op::OpKernelState {
 public:
  GPUPoolOpKernelState(const int32_t dim, const std::string& pooling_type, const Shape& x_shape,
                       const Shape& y_shape, const std::string& data_format, const DataType& dtype,
                       const Params3D& params_3d)
      : is_dynamic_(false), dim_(dim), pooling_type_(pooling_type) {
    Reset(dim, pooling_type, x_shape, y_shape, data_format, dtype, params_3d);
  }
  GPUPoolOpKernelState(const int32_t dim, const std::string& pooling_type)
      : is_dynamic_(true), dim_(dim), pooling_type_(pooling_type) {}
  ~GPUPoolOpKernelState() = default;

  void ResetIfDynamic(user_op::KernelComputeContext* ctx) {
    if (this->is_dynamic_) {
      const ShapeView& x_shape = ctx->Tensor4ArgNameAndIndex("x", 0)->shape();
      const std::string data_format = ctx->GetAttr<std::string>("data_format");
      const std::string padding = ctx->GetAttr<std::string>("padding");
      const std::vector<int32_t>& pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
      const std::vector<int32_t>& strides = ctx->GetAttr<std::vector<int32_t>>("strides");
      const Params3D params_3d(dim_, x_shape, data_format, padding, pool_size, strides);
      const ShapeView& y_shape = ctx->Tensor4ArgNameAndIndex("y", 0)->shape();
      const DataType dtype = ctx->Tensor4ArgNameAndIndex("x", 0)->data_type();
      Reset(dim_, pooling_type_, x_shape, y_shape, data_format, dtype, params_3d);
    }
  }

  void Reset(const int32_t dim, const std::string& pooling_type, const ShapeView& x_shape,
             const ShapeView& y_shape, const std::string& data_format, const DataType& dtype,
             const Params3D& params_3d) {
    FixedVector pool_size(dim);
    FixedVector padding(dim);
    FixedVector strides(dim);
    FOR_RANGE(int, i, 0, dim) {
      int32_t index_in_3d = i + 3 - dim;
      pool_size[i] = params_3d.pool_size_3d().at(index_in_3d);
      padding[i] = std::max<int>(params_3d.padding_before_3d().at(index_in_3d),
                                 params_3d.padding_after_3d().at(index_in_3d));
      strides[i] = params_3d.strides_3d().at(index_in_3d);
    }

    x_desc_.reset(new CudnnTensorDesc(dtype, x_shape, data_format));
    y_desc_.reset(new CudnnTensorDesc(dtype, y_shape, data_format));
    cudnnPoolingMode_t pooling_mode;
    if (pooling_type == "AVG") {
      pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else if (pooling_type == "MAX") {
      pooling_mode = CUDNN_POOLING_MAX;
    } else {
      UNIMPLEMENTED();
    }
    pooling_desc_.reset(
        new CudnnPoolDesc(pooling_mode, dim, pool_size.data(), padding.data(), strides.data()));
  }

  static std::shared_ptr<user_op::OpKernelState> FromKernelInitContext(
      const int32_t& dim, const std::string& pooling_type, user_op::KernelInitContext* ctx) {
    if (pooling_type != "MAX" && pooling_type != "AVG") { UNIMPLEMENTED(); }
    std::shared_ptr<GPUPoolOpKernelState> state;
    const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
    if (x_desc->is_dynamic()) {
      state.reset(new GPUPoolOpKernelState(dim, pooling_type));
    } else {
      const Shape& x_shape = x_desc->shape();
      const std::string data_format = ctx->GetAttr<std::string>("data_format");
      const std::string padding = ctx->GetAttr<std::string>("padding");
      const std::vector<int32_t>& pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
      const std::vector<int32_t>& strides = ctx->GetAttr<std::vector<int32_t>>("strides");
      const Params3D params_3d(dim, x_shape, data_format, padding, pool_size, strides);
      const Shape y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();
      const DataType dtype = x_desc->data_type();
      state.reset(new GPUPoolOpKernelState(dim, pooling_type, x_shape, y_shape, data_format, dtype,
                                           params_3d));
    }
    return std::move(state);
  }

  const cudnnTensorDescriptor_t& cudnn_x_tensor_desc() const { return x_desc_->Get(); }
  const cudnnTensorDescriptor_t& cudnn_y_tensor_desc() const { return y_desc_->Get(); }
  const cudnnPoolingDescriptor_t& cudnn_pooling_desc() const { return pooling_desc_->Get(); }

 private:
  std::unique_ptr<CudnnTensorDesc> x_desc_;
  std::unique_ptr<CudnnTensorDesc> y_desc_;
  std::unique_ptr<CudnnPoolDesc> pooling_desc_;
  bool is_dynamic_;
  int32_t dim_;
  std::string pooling_type_;
};

template<typename dtype>
std::function<bool(const user_op::KernelRegContext& ctx)> MakeIsMatchedPred(
    DeviceType device_type) {
  return [device_type](const user_op::KernelRegContext& ctx) {
    const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
    return ctx.device_type() == device_type && y_desc->data_type() == GetDataType<dtype>::value;
  };
}

template<typename T>
struct PoolGpuKernelUtil {
  static void FWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    GPUPoolOpKernelState* gpu_pool_op_kernel_state = dynamic_cast<GPUPoolOpKernelState*>(state);
    gpu_pool_op_kernel_state->ResetIfDynamic(ctx);
    CHECK(gpu_pool_op_kernel_state != nullptr);
    CudaCheck(cudnnPoolingForward(
        ctx->device_ctx()->cudnn_handle(), gpu_pool_op_kernel_state->cudnn_pooling_desc(),
        CudnnSPOnePtr<T>(), gpu_pool_op_kernel_state->cudnn_x_tensor_desc(), x->dptr(),
        CudnnSPZeroPtr<T>(), gpu_pool_op_kernel_state->cudnn_y_tensor_desc(), y->mut_dptr()));
  }

  static void BWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    GPUPoolOpKernelState* gpu_pool_op_kernel_state = dynamic_cast<GPUPoolOpKernelState*>(state);
    gpu_pool_op_kernel_state->ResetIfDynamic(ctx);
    CHECK(gpu_pool_op_kernel_state != nullptr);
    CudaCheck(cudnnPoolingBackward(
        ctx->device_ctx()->cudnn_handle(), gpu_pool_op_kernel_state->cudnn_pooling_desc(),
        CudnnSPOnePtr<T>(), gpu_pool_op_kernel_state->cudnn_y_tensor_desc(), y->dptr(),
        gpu_pool_op_kernel_state->cudnn_y_tensor_desc(), dy->dptr(),
        gpu_pool_op_kernel_state->cudnn_x_tensor_desc(), x->dptr(), CudnnSPZeroPtr<T>(),
        gpu_pool_op_kernel_state->cudnn_x_tensor_desc(), dx->mut_dptr()));
  }
};

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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::FWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::BWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::FWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::BWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::FWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::BWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::FWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::BWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::FWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::BWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::FWCompute(ctx, state);
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
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolGpuKernelUtil<T>::BWCompute(ctx, state);
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
REGISTER_POOL_GPU_KERNEL(float16)

}  // namespace oneflow
