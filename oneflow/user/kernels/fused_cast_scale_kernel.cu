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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {

template<typename T, typename U>
__global__ void FusedCastScaleGpu(const int64_t n, const T scale_val, const U* in,
                                  const T* scale_by_ptr, T* out) {
  const T scale = *scale_by_ptr * scale_val;
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = static_cast<T>(in[i]) * scale; }
}

template<>
__global__ void FusedCastScaleGpu<float, half>(const int64_t n, const float scale_val,
                                               const half* in, const float* scale_by_ptr,
                                               float* out) {
  const float scale = *scale_by_ptr * scale_val;
  const int64_t n_2 = n / 2;
  const auto* in_2 = reinterpret_cast<const half2*>(in);
  auto* out_2 = reinterpret_cast<float2*>(out);
  CUDA_1D_KERNEL_LOOP(i, n_2) {
    float2 f2 = __half22float2(in_2[i]);
    f2.x *= scale;
    f2.y *= scale;
    out_2[i] = f2;
  }
  if (n % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[n - 1] = __half2float(in[n - 1]) * scale;
  }
}

template<>
__global__ void FusedCastScaleGpu<half, float>(const int64_t n, const half scale_val,
                                               const float* in, const half* scale_by_ptr,
                                               half* out) {
  const half scale = *scale_by_ptr * scale_val;
  const half2 scale_h2 = __half2half2(scale);
  const int64_t n_2 = n / 2;
  const auto* in_2 = reinterpret_cast<const float2*>(in);
  auto* out_h2 = reinterpret_cast<half2*>(out);
  CUDA_1D_KERNEL_LOOP(i, n_2) {
    half2 in_h2 = __float22half2_rn(in_2[i]);
    out_h2[i] = __hmul2(in_h2, scale_h2);
  }
  if (n % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[n - 1] = __float2half(in[n - 1]) * scale;
  }
}

#if CUDA_VERSION >= 11000 && __CUDA_ARCH__ >= 800
template<>
__global__ void FusedCastScaleGpu<float, nv_bfloat16>(const int64_t n, const float scale_val,
                                                      const nv_bfloat16* in,
                                                      const float* scale_by_ptr, float* out) {
  const float scale = *scale_by_ptr * scale_val;
  const int64_t n_2 = n / 2;
  const auto* in_2 = reinterpret_cast<const nv_bfloat162*>(in);
  auto* out_2 = reinterpret_cast<float2*>(out);
  CUDA_1D_KERNEL_LOOP(i, n_2) {
    float2 f2 = __bfloat1622float2(in_2[i]);
    f2.x *= scale;
    f2.y *= scale;
    out_2[i] = f2;
  }
  if (n % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[n - 1] = __bfloat162float(in[n - 1]) * scale;
  }
}

template<>
__global__ void FusedCastScaleGpu<nv_bfloat16, float>(const int64_t n, const nv_bfloat16 scale_val,
                                                      const float* in,
                                                      const nv_bfloat16* scale_by_ptr,
                                                      nv_bfloat16* out) {
  const nv_bfloat16 scale = *scale_by_ptr * scale_val;
  const nv_bfloat162 scale_h2 = __bfloat162bfloat162(scale);
  const int64_t n_2 = n / 2;
  const auto* in_2 = reinterpret_cast<const float2*>(in);
  auto* out_h2 = reinterpret_cast<nv_bfloat162*>(out);
  CUDA_1D_KERNEL_LOOP(i, n_2) {
    nv_bfloat162 in_h2 = __float22bfloat162_rn(in_2[i]);
    out_h2[i] = __hmul2(in_h2, scale_h2);
  }
  if (n % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[n - 1] = __float2bfloat16(in[n - 1]) * scale;
  }
}
#endif

template<typename T, typename U>
class FusedCastScaleGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  FusedCastScaleGpuKernel() = default;
  ~FusedCastScaleGpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t n = x->shape_view().elem_cnt();
    const double scale = ctx->Attr<double>("scale");
    const bool use_pack =
        (x->data_type() == DataType::kFloat
         && (y->data_type() == DataType::kFloat16 || y->data_type() == DataType::kBFloat16))
        || (y->data_type() == DataType::kFloat
            && (x->data_type() == DataType::kFloat16 || x->data_type() == DataType::kBFloat16));
    const int64_t launch_n = use_pack ? RoundUp(n, 2) / 2 : n;
    FusedCastScaleGpu<T, U><<<BlocksNum4ThreadsNum(launch_n), kCudaThreadsNumPerBlock, 0,
                              ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        n, static_cast<T>(scale), x->dptr<U>(), scale_by_tensor->dptr<T>(), y->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(x_type, y_type)                          \
  REGISTER_USER_KERNEL("fused_cast_scale")                                             \
      .SetCreateFn<FusedCastScaleGpuKernel<y_type, x_type>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("y", 0) == GetDataType<y_type>::value) \
                       && (user_op::HobDataType("x", 0) == GetDataType<x_type>::value));

REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(half, float);
REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(half, double);
REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(float, half);
REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(float, double);
REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(double, half);
REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(double, float);
#if CUDA_VERSION >= 11000
REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(nv_bfloat16, float);
REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL(float, nv_bfloat16);
#endif
#undef REGISTER_FUSED_CAST_SCALE_CUDA_KERNEL

}  // namespace oneflow
