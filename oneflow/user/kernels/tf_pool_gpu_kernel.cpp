/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifdef WITH_CUDA

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/utils/pool_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

class CudnnPoolDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolDesc);
  CudnnPoolDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window, const int* padding,
                const int* stride) {
    OF_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&val_));
    OF_CUDNN_CHECK(cudnnSetPoolingNdDescriptor(val_, pooling_mode, CUDNN_NOT_PROPAGATE_NAN, dims,
                                               window, padding, stride));
  }

  ~CudnnPoolDesc() { OF_CUDNN_CHECK(cudnnDestroyPoolingDescriptor(val_)); }

  const cudnnPoolingDescriptor_t& Get() const { return val_; }

 private:
  cudnnPoolingDescriptor_t val_;
};

class GPUPoolOpKernelCache final : public user_op::OpKernelCache {
 public:
  GPUPoolOpKernelCache(const int32_t dim, const std::string& pooling_type, const ShapeView& x_shape,
                       const ShapeView& y_shape, const std::string& data_format,
                       const DataType& dtype, const Params3D& params_3d)
      : pooling_type_(pooling_type) {
    Reset(dim, pooling_type, x_shape, y_shape, data_format, dtype, params_3d);
  }
  ~GPUPoolOpKernelCache() = default;

  void Reset(const int32_t dim, const std::string& pooling_type, const ShapeView& x_shape,
             const ShapeView& y_shape, const std::string& data_format, const DataType& dtype,
             const Params3D& params_3d) {
    FixedVector pool_size(dim);
    FixedVector padding(dim);
    FixedVector strides(dim);
    FOR_RANGE(int, i, 0, dim) {
      int32_t index_in_3d = i + 3 - dim;
      pool_size[i] = params_3d.pool_size_3d().at(index_in_3d);
      padding[i] = params_3d.padding_before_3d().at(index_in_3d);
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

  static std::shared_ptr<GPUPoolOpKernelCache> FromKernelComputeContext(
      const int32_t& dim, const std::string& pooling_type, user_op::KernelCacheContext* ctx) {
    if (pooling_type != "MAX" && pooling_type != "AVG") { UNIMPLEMENTED(); }
    const ShapeView& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& padding = ctx->Attr<std::string>("padding");
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t>& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    const Params3D params_3d(dim, x_shape, data_format, padding, padding_before, padding_after,
                             pool_size, strides, ceil_mode);
    const ShapeView& y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();
    const DataType dtype = ctx->TensorDesc4ArgNameAndIndex("x", 0)->data_type();
    return std::make_shared<GPUPoolOpKernelCache>(dim, pooling_type, x_shape, y_shape, data_format,
                                                  dtype, params_3d);
  }

  const cudnnTensorDescriptor_t& cudnn_x_tensor_desc() const { return x_desc_->Get(); }
  const cudnnTensorDescriptor_t& cudnn_y_tensor_desc() const { return y_desc_->Get(); }
  const cudnnPoolingDescriptor_t& cudnn_pooling_desc() const { return pooling_desc_->Get(); }

 private:
  std::unique_ptr<CudnnTensorDesc> x_desc_;
  std::unique_ptr<CudnnTensorDesc> y_desc_;
  std::unique_ptr<CudnnPoolDesc> pooling_desc_;
  std::string pooling_type_;
};

struct PoolGpuKernelUtil {
  static void FWCompute(user_op::KernelComputeContext* ctx,
                        const GPUPoolOpKernelCache* gpu_pool_op_kernel_cache) {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    CHECK(gpu_pool_op_kernel_cache != nullptr);
    OF_CUDNN_CHECK(cudnnPoolingForward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(),
        gpu_pool_op_kernel_cache->cudnn_pooling_desc(), CudnnSPOnePtr(x->data_type()),
        gpu_pool_op_kernel_cache->cudnn_x_tensor_desc(), x->dptr(), CudnnSPZeroPtr(x->data_type()),
        gpu_pool_op_kernel_cache->cudnn_y_tensor_desc(), y->mut_dptr()));
  }

  static void BWCompute(user_op::KernelComputeContext* ctx,
                        const GPUPoolOpKernelCache* gpu_pool_op_kernel_cache) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    CHECK(gpu_pool_op_kernel_cache != nullptr);
    OF_CUDNN_CHECK(cudnnPoolingBackward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(),
        gpu_pool_op_kernel_cache->cudnn_pooling_desc(), CudnnSPOnePtr(y->data_type()),
        gpu_pool_op_kernel_cache->cudnn_y_tensor_desc(), y->dptr(),
        gpu_pool_op_kernel_cache->cudnn_y_tensor_desc(), dy->dptr(),
        gpu_pool_op_kernel_cache->cudnn_x_tensor_desc(), x->dptr(), CudnnSPZeroPtr(y->data_type()),
        gpu_pool_op_kernel_cache->cudnn_x_tensor_desc(), dx->mut_dptr()));
  }
};

}  // namespace

class AvgPool1DGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AvgPool1DGpuKernel() = default;
  ~AvgPool1DGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(1, "AVG", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::FWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class AvgPool1DGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AvgPool1DGradGpuKernel() = default;
  ~AvgPool1DGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(1, "AVG", ctx);
  }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::BWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class AvgPool2DGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AvgPool2DGpuKernel() = default;
  ~AvgPool2DGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(2, "AVG", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::FWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class AvgPool2DGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AvgPool2DGradGpuKernel() = default;
  ~AvgPool2DGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(2, "AVG", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::BWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class AvgPool3DGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AvgPool3DGpuKernel() = default;
  ~AvgPool3DGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(3, "AVG", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::FWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class AvgPool3DGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AvgPool3DGradGpuKernel() = default;
  ~AvgPool3DGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(3, "AVG", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::BWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class MaxPool1DGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MaxPool1DGpuKernel() = default;
  ~MaxPool1DGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(1, "MAX", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::FWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class MaxPool1DGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MaxPool1DGradGpuKernel() = default;
  ~MaxPool1DGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(1, "MAX", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::BWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class MaxPool2DGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MaxPool2DGpuKernel() = default;
  ~MaxPool2DGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(2, "MAX", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::FWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class MaxPool2DGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MaxPool2DGradGpuKernel() = default;
  ~MaxPool2DGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(2, "MAX", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::BWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class MaxPool3DGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MaxPool3DGpuKernel() = default;
  ~MaxPool3DGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(3, "MAX", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::FWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

class MaxPool3DGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MaxPool3DGradGpuKernel() = default;
  ~MaxPool3DGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return GPUPoolOpKernelCache::FromKernelComputeContext(3, "MAX", ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    PoolGpuKernelUtil::BWCompute(ctx, dynamic_cast<const GPUPoolOpKernelCache*>(cache));
  };
};

REGISTER_USER_KERNEL("tf_avg_pool_1d")
    .SetCreateFn<AvgPool1DGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_avg_pool_1d_grad")
    .SetCreateFn<AvgPool1DGradGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_avg_pool_2d")
    .SetCreateFn<AvgPool2DGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_avg_pool_2d_grad")
    .SetCreateFn<AvgPool2DGradGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_avg_pool_3d")
    .SetCreateFn<AvgPool3DGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_avg_pool_3d_grad")
    .SetCreateFn<AvgPool3DGradGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_max_pool_1d")
    .SetCreateFn<MaxPool1DGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_max_pool_1d_grad")
    .SetCreateFn<MaxPool1DGradGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_max_pool_2d")
    .SetCreateFn<MaxPool2DGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_max_pool_2d_grad")
    .SetCreateFn<MaxPool2DGradGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_max_pool_3d")
    .SetCreateFn<MaxPool3DGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));
REGISTER_USER_KERNEL("tf_max_pool_3d_grad")
    .SetCreateFn<MaxPool3DGradGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));

}  // namespace oneflow

#endif
