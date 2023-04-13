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
#include <cub/cub.cuh>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/cuda/layer_norm.cuh"

namespace oneflow {

namespace {

template<typename T, typename ComputeType>
__global__ void L2NormalizeForward(const int32_t n, const int32_t c, const int32_t d,
                                   const ComputeType epsilon, const T* in,
                                   ComputeType* square_x_sum, T* out) {
  using BlockReduce = cub::BlockReduce<ComputeType, ep::CudaStream::kDefaultBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    ComputeType sum = GetZeroVal<ComputeType>();
    const int32_t offset = (i / d) * d * c + (i % d);
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const ComputeType x = static_cast<ComputeType>(in[offset + j * d]);
      sum += x * x;
    }
    const ComputeType reduce_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) { square_x_sum[i] = reduce_sum; }
    __syncthreads();

    const ComputeType inv_norm = rsqrtf(fmaxf(square_x_sum[i], epsilon));
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const int32_t index = offset + j * d;
      out[index] = static_cast<T>(inv_norm * static_cast<ComputeType>(in[index]));
    }
  }
}

template<typename T>
__global__ void L2NormalizeBackward(const int32_t n, const int32_t c, const int32_t d,
                                    const float epsilon, const T* out, const T* out_diff,
                                    const T* square_x_sum, T* in_diff) {
  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    const T inv_norm = rsqrt(fmaxf(square_x_sum[i], epsilon));
    const int32_t offset = (i / d) * d * c + (i % d);
    if (square_x_sum[i] >= epsilon) {
      using BlockReduce = cub::BlockReduce<T, ep::CudaStream::kDefaultBlockSize>;
      __shared__ typename BlockReduce::TempStorage temp_storage_prod_sum;

      T y_dy_prod_sum = GetZeroVal<T>();
      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        y_dy_prod_sum += out[index] * out_diff[index];
      }

      const T reduce_y_dy_prod_sum = BlockReduce(temp_storage_prod_sum).Sum(y_dy_prod_sum);
      __shared__ T y_dy_inner_prod;
      if (threadIdx.x == 0) { y_dy_inner_prod = reduce_y_dy_prod_sum; }
      __syncthreads();

      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        in_diff[index] = inv_norm * (out_diff[index] - y_dy_inner_prod * out[index]);
      }
    } else {
      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        in_diff[index] = inv_norm * out_diff[index];
      }
    }
  }
}

}  // namespace

template<typename T>
class GpuL2NormalizeKernel final : public user_op::OpKernel {
 public:
  GpuL2NormalizeKernel() = default;
  ~GpuL2NormalizeKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* square_x_sum = ctx->Tensor4ArgNameAndIndex("square_x_sum", 0);
    const float epsilon = ctx->Attr<float>("epsilon");
    int32_t axis = ctx->Attr<int32_t>("axis");
    int32_t c = x->shape_view().At(axis);
    int32_t n = x->shape_view().elem_cnt() / c;
    int32_t d = x->shape_view().Count(axis + 1);
    using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
    RUN_CUDA_KERNEL((L2NormalizeForward<T, ComputeType>), ctx->stream(), n, n, c, d,
                    static_cast<ComputeType>(epsilon), x->dptr<T>(),
                    square_x_sum->mut_dptr<ComputeType>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_L2_NORMALIZE_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("l2_normalize")                                 \
      .SetCreateFn<GpuL2NormalizeKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_L2_NORMALIZE_KERNEL(half)
REGISTER_CUDA_L2_NORMALIZE_KERNEL(float)
REGISTER_CUDA_L2_NORMALIZE_KERNEL(double)

template<typename T>
class GpuL2NormalizeGradKernel final : public user_op::OpKernel {
 public:
  GpuL2NormalizeGradKernel() = default;
  ~GpuL2NormalizeGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* square_x_sum = ctx->Tensor4ArgNameAndIndex("square_x_sum", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float epsilon = ctx->Attr<float>("epsilon");
    int32_t axis = ctx->Attr<int32_t>("axis");
    int32_t c = dy->shape_view().At(axis);
    int32_t n = dy->shape_view().elem_cnt() / c;
    int32_t d = dy->shape_view().Count(axis + 1);
    RUN_CUDA_KERNEL((L2NormalizeBackward<T>), ctx->stream(), n, n, c, d, static_cast<T>(epsilon),
                    y->dptr<T>(), dy->dptr<T>(), square_x_sum->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_L2_NORMALIZE_GRAD_KERNEL(dtype)                  \
  REGISTER_USER_KERNEL("l2_normalize_grad")                            \
      .SetCreateFn<GpuL2NormalizeGradKernel<dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_L2_NORMALIZE_GRAD_KERNEL(float)
REGISTER_CUDA_L2_NORMALIZE_GRAD_KERNEL(double)

}  // namespace oneflow
