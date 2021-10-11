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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/layer_norm.cuh"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename SRC, typename DST>
struct ScaleCenterStore {
  ScaleCenterStore(DST* normalized, DST* y, int64_t row_size, const DST* gamma, const DST* beta)
      : normalized(normalized), y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, N> y_pack;
    cuda::layer_norm::Pack<DST, N> normalized_pack;
    cuda::layer_norm::Pack<DST, N> gamma_pack;
    cuda::layer_norm::Pack<DST, N> beta_pack;
    const int64_t offset = row * row_size + col;
    // TODO if gamma beta nullptr
    if (gamma != nullptr) {
      gamma_pack.storage =
          *reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(gamma + col);
    } else {
      for (int i = 0; i < N; ++i) { gamma_pack.elem[i] = 1; }
    }
    if (beta != nullptr) {
      beta_pack.storage = *reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(beta + col);
    } else {
      for (int i = 0; i < N; ++i) { beta_pack.elem[i] = 0; }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      normalized_pack.elem[i] = normalized_i;
      y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
    }
    *reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(y + offset) = y_pack.storage;
    if (gamma != nullptr) {
      *reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(normalized + offset) =
          normalized_pack.storage;
    }
  }
  DST* normalized;
  DST* y;
  int64_t row_size;
  const DST* gamma;
  const DST* beta;
};

template<typename T>
void LayerNormForwardGpu(DeviceCtx* ctx, const int num_instances, const int norm_size,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* normalized_ptr, T* y_ptr, user_op::Tensor* mean,
                         user_op::Tensor* inv_variance) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, ComputeType> load(x_ptr, norm_size);
  ScaleCenterStore<ComputeType, T> store(normalized_ptr, y_ptr, norm_size, gamma_ptr, beta_ptr);
  cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
      ctx->cuda_stream(), load, store, num_instances, norm_size, epsilon,
      mean->mut_dptr<ComputeType>(), inv_variance->mut_dptr<ComputeType>());
}

}  // namespace

template<typename T>
class LayerNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LayerNormGpuKernel() = default;
  ~LayerNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
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
    int32_t instance_size = 0;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (scale || center) {
      if (scale) {
        const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
        instance_size = gamma->shape().elem_cnt();
        gamma_ptr = gamma->dptr<T>();
      }
      if (center) {
        const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
        if (gamma_ptr) {
          CHECK_EQ(beta->shape().elem_cnt(), instance_size);
        } else {
          instance_size = beta->shape().elem_cnt();
        }
        beta_ptr = beta->dptr<T>();
      }
      CHECK_EQ(y->shape().elem_cnt() % instance_size, 0);
    }
    LayerNormForwardGpu<T>(ctx->device_ctx(), num_instances, norm_size, epsilon, x->dptr<T>(),
                           gamma_ptr, beta_ptr, normalized->mut_dptr<T>(), y->mut_dptr<T>(), mean,
                           inv_variance);
  }
};

#define REGISTER_LAYER_NORM_GPU_KERNEL(dtype)             \
  REGISTER_USER_KERNEL("layer_norm")                      \
      .SetCreateFn<LayerNormGpuKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_GPU_KERNEL(half)
REGISTER_LAYER_NORM_GPU_KERNEL(float)
REGISTER_LAYER_NORM_GPU_KERNEL(double)

}  // namespace oneflow
