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
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename U>
__global__ void FusedCastScaleGpu(const int64_t n, const U* in, const T* scalar, T* out) {
  const T scalar_val = *scalar;
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = static_cast<T>(in[i]) * scalar_val; }
}

template<>
__global__ void FusedCastScaleGpu<float, half>(const int64_t n, const half* in, const float* scalar,
                                               float* out) {
  const float scalar_val = *scalar;
  const int64_t n_2 = n / 2;
  const auto* in_2 = reinterpret_cast<const half2*>(in);
  auto* out_2 = reinterpret_cast<float2*>(out);
  CUDA_1D_KERNEL_LOOP(i, n_2) {
    float2 f2 = __half22float2(in_2[i]);
    f2.x *= scalar_val;
    f2.y *= scalar_val;
    out_2[i] = f2;
  }
  if (n % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[n - 1] = __half2float(in[n - 1]) * scalar_val;
  }
}

template<>
__global__ void FusedCastScaleGpu<half, float>(const int64_t n, const float* in, const half* scalar,
                                               half* out) {
  const half scalar_val = *scalar;
  const half2 scalar_h2 = __half2half2(scalar_val);
  const int64_t n_2 = n / 2;
  const auto* in_2 = reinterpret_cast<const float2*>(in);
  auto* out_h2 = reinterpret_cast<half2*>(out);
  CUDA_1D_KERNEL_LOOP(i, n_2) {
    half2 in_h2 = __float22half2_rn(in_2[i]);
    out_h2[i] = __hmul2(in_h2, scalar_h2);
  }
  if (n % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[n - 1] = __float2half(in[n - 1]) * scalar_val;
  }
}

template<typename T, typename U>
class FusedCastScaleGpuKernel final : public user_op::OpKernel {
 public:
  FusedCastScaleGpuKernel() = default;
  ~FusedCastScaleGpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t n = x->shape().elem_cnt();
    const int64_t launch_n = ((std::is_same<T, half>::value && std::is_same<U, float>::value)
                              || (std::is_same<T, float>::value && std::is_same<U, half>::value))
                                 ? RoundUp(n, 2) / 2
                                 : n;
    FusedCastScaleGpu<T, U><<<BlocksNum4ThreadsNum(launch_n), kCudaThreadsNumPerBlock, 0,
                              ctx->device_ctx()->cuda_stream()>>>(
        n, x->dptr<U>(), scalar->dptr<T>(), y->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_FUSED_CAST_SCALE_GPU_KERNEL(x_type, y_type)                          \
  REGISTER_USER_KERNEL("fused_cast_scale")                                            \
      .SetCreateFn<FusedCastScaleGpuKernel<y_type, x_type>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                             \
                       & (user_op::HobDataType("y", 0) == GetDataType<y_type>::value) \
                       & (user_op::HobDataType("x", 0) == GetDataType<x_type>::value));

REGISTER_FUSED_CAST_SCALE_GPU_KERNEL(half, float);
REGISTER_FUSED_CAST_SCALE_GPU_KERNEL(half, double);
REGISTER_FUSED_CAST_SCALE_GPU_KERNEL(float, half);
REGISTER_FUSED_CAST_SCALE_GPU_KERNEL(float, double);
REGISTER_FUSED_CAST_SCALE_GPU_KERNEL(double, half);
REGISTER_FUSED_CAST_SCALE_GPU_KERNEL(double, float);
#undef REGISTER_FUSED_CAST_SCALE_GPU_KERNEL

}  // namespace oneflow
