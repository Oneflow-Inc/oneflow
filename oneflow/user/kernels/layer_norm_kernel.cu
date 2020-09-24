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

#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int64_t kLayerNormGpuBlockSize = 256;

template<typename T>
struct LayerNormUtil {
  using ComputeType = T;
  __device__ static ComputeType ToComputeType(T v) { return v; }
  __device__ static T FromComputeType(ComputeType v) { return v; }
};

template<>
struct LayerNormUtil<half> {
  using ComputeType = float;
  __device__ static ComputeType ToComputeType(half v) { return __half2float(v); }
  __device__ static half FromComputeType(ComputeType v) { return __float2half(v); }
};

template<typename T>
int GetForwardDynamicSharedMemorySize(const int norm_size) {
  return norm_size * sizeof(typename LayerNormUtil<T>::ComputeType);
}

int GetLayerNormBlockSize() { return kLayerNormGpuBlockSize; }

int GetLayerNormNumBlocks(const int num_instances) {
  return std::min(static_cast<int>(num_instances), kCudaMaxBlocksNum);
}

template<typename T, typename ComputeType>
__global__ void LayerNormImpl(const int num_instances, const int norm_size, const int params_size,
                              const bool do_scale, const bool do_center, const double epsilon,
                              const T* x, const T* gamma, const T* beta, ComputeType* mean,
                              ComputeType* inv_variance, T* normalized, T* y) {
  using LU = LayerNormUtil<T>;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char fw_shared_buf[];
  auto* compute_buf = reinterpret_cast<ComputeType*>(fw_shared_buf);
  __shared__ ComputeType row_reduce_mean;
  __shared__ ComputeType row_reduce_inv_var;
  typedef cub::BlockReduce<ComputeType, kLayerNormGpuBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  ComputeType val_scale = ComputeType(1.0) / static_cast<ComputeType>(norm_size);
  for (int row = blockIdx.x; row < num_instances; row += gridDim.x) {
    const int row_offset = row * norm_size;
    const T* in_row = x + row_offset;
    ComputeType thread_sum = 0;
    ComputeType thread_square_sum = 0;
    const int tid = threadIdx.x;
    for (int col = tid; col < norm_size; col += blockDim.x) {
      const ComputeType val = LU::ToComputeType(in_row[col]);
      compute_buf[col] = val;
      thread_sum += val;
      thread_square_sum += val * val;
    }
    __syncthreads();
    ComputeType block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
    ComputeType block_square_sum =
        BlockReduce(cub_reduce_tmp_storage).Reduce(thread_square_sum, cub::Sum());
    if (tid == 0) {
      ComputeType mean_val = block_sum * val_scale;
      row_reduce_mean = mean_val;
      mean[row] = mean_val;
      ComputeType variance_val =
          max(block_square_sum * val_scale - mean_val * mean_val, ComputeType(0));
      ComputeType inv_var_val = rsqrt(variance_val + static_cast<ComputeType>(epsilon));
      row_reduce_inv_var = inv_var_val;
      inv_variance[row] = inv_var_val;
    }
    __syncthreads();
    ComputeType mean = row_reduce_mean;
    ComputeType inv_var = row_reduce_inv_var;
    for (int col = threadIdx.x; col < norm_size; col += blockDim.x) {
      int offset = row_offset + col;
      ComputeType val = compute_buf[col];
      val = (val - mean) * inv_var;
      if (do_scale || do_center) {
        int elem_id = offset % params_size;
        if (do_scale) {
          normalized[offset] = LU::FromComputeType(val);
          val *= LU::ToComputeType(gamma[elem_id]);
        }
        if (do_center) { val += LU::ToComputeType(beta[elem_id]); }
      }
      y[offset] = LU::FromComputeType(val);
    }
  }
}

template<typename T>
void LayerNormForwardGpu(DeviceCtx* ctx, const int num_instances, const int norm_size,
                         const int params_size, const bool scale, const bool center,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* normalized_ptr, T* y_ptr, user_op::Tensor* mean,
                         user_op::Tensor* inv_variance) {
  LayerNormImpl<T, typename LayerNormUtil<T>::ComputeType>
      <<<GetLayerNormNumBlocks(num_instances), GetLayerNormBlockSize(),
         GetForwardDynamicSharedMemorySize<T>(norm_size), ctx->cuda_stream()>>>(
          num_instances, norm_size, params_size, scale, center, epsilon, x_ptr, gamma_ptr, beta_ptr,
          mean->mut_dptr<typename LayerNormUtil<T>::ComputeType>(),
          inv_variance->mut_dptr<typename LayerNormUtil<T>::ComputeType>(), normalized_ptr, y_ptr);
}

template<>
void LayerNormForwardGpu<float16>(DeviceCtx* ctx, const int num_instances, const int norm_size,
                                  const int params_size, const bool scale, const bool center,
                                  const double epsilon, const float16* x_ptr,
                                  const float16* gamma_ptr, const float16* beta_ptr,
                                  float16* normalized_ptr, float16* y_ptr, user_op::Tensor* mean,
                                  user_op::Tensor* inv_variance) {
  LayerNormImpl<half, typename LayerNormUtil<half>::ComputeType>
      <<<GetLayerNormNumBlocks(num_instances), GetLayerNormBlockSize(),
         GetForwardDynamicSharedMemorySize<half>(norm_size), ctx->cuda_stream()>>>(
          num_instances, norm_size, params_size, scale, center, epsilon,
          reinterpret_cast<const half*>(x_ptr), reinterpret_cast<const half*>(gamma_ptr),
          reinterpret_cast<const half*>(beta_ptr),
          mean->mut_dptr<typename LayerNormUtil<half>::ComputeType>(),
          inv_variance->mut_dptr<typename LayerNormUtil<half>::ComputeType>(),
          reinterpret_cast<half*>(normalized_ptr), reinterpret_cast<half*>(y_ptr));
}

}  // namespace

template<typename T>
class LayerNormFuseKernel final : public user_op::OpKernel {
 public:
  LayerNormFuseKernel() = default;
  ~LayerNormFuseKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const bool scale = ctx->Attr<bool>("scale");
    const bool center = ctx->Attr<bool>("center");
    user_op::Tensor* normalized = scale ? ctx->Tensor4ArgNameAndIndex("normalized", 0) : y;
    const double epsilon = ctx->Attr<double>("epsilon");
    const int32_t num_instances = mean->shape().elem_cnt();
    const int32_t norm_size = x->shape().elem_cnt() / num_instances;
    int32_t params_size = 0;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (scale || center) {
      if (scale) {
        const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
        params_size = gamma->shape().elem_cnt();
        gamma_ptr = gamma->dptr<T>();
      }
      if (center) {
        const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
        if (gamma_ptr) {
          CHECK_EQ(beta->shape().elem_cnt(), params_size);
        } else {
          params_size = beta->shape().elem_cnt();
        }
        beta_ptr = beta->dptr<T>();
      }
    }
    LayerNormForwardGpu<T>(ctx->device_ctx(), num_instances, norm_size, params_size, scale, center,
                           epsilon, x->dptr<T>(), gamma_ptr, beta_ptr, normalized->mut_dptr<T>(),
                           y->mut_dptr<T>(), mean, inv_variance);
  }
};

#define REGISTER_LAYER_NORM_FUSE_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("layer_norm")                                                 \
      .SetCreateFn<LayerNormFuseKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                            \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobAttr<bool>("for_test") == true));

REGISTER_LAYER_NORM_FUSE_KERNEL(float)
REGISTER_LAYER_NORM_FUSE_KERNEL(double)
REGISTER_LAYER_NORM_FUSE_KERNEL(float16)

}  // namespace oneflow
