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

__global__ void FusedCastScaleGpu(const int64_t n, const half* in, const float* scalar,
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

template<DeviceType device, typename T, typename U>
class FusedCastScaleKernel final : public user_op::OpKernel {
 public:
  FusedCastScaleKernel() = default;
  ~FusedCastScaleKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t n = x->shape().elem_cnt();
    FusedCastScaleGpu<<<BlocksNum4ThreadsNum(RoundUp(n, 2) / 2), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(n, x->dptr<U>(), scalar->dptr<T>(),
                                                            y->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_KERNEL(device_type, x_type, y_type)                                  \
  REGISTER_USER_KERNEL("fused_cast_scale")                                            \
      .SetCreateFn<FusedCastScaleKernel<device_type, y_type, x_type>>()               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type)                       \
                       & (user_op::HobDataType("y", 0) == GetDataType<y_type>::value) \
                       & (user_op::HobDataType("x", 0) == GetDataType<x_type>::value));

REGISTER_KERNEL(DeviceType::kGPU, half, float);

}  // namespace oneflow
