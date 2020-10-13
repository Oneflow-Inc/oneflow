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
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void MaskAndScaleGpu(const int64_t n, float scale, const T* x, const int8_t* mask,
                                T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
}

template<typename T>
__global__ void MaskAndScaleAddGpu(const int64_t n, float scale, const T* x, const int8_t* mask,
                                   const T* addend, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * static_cast<T>(mask[i]) * scale + addend[i]; }
}

template<>
__global__ void MaskAndScaleGpu<half>(const int64_t n, float scale, const half* x,
                                      const int8_t* mask, half* y) {
  const int64_t h2_n = n / 2;
  half2 h2_scale = __float2half2_rn(scale);
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  const auto* mask_c2 = reinterpret_cast<const char2*>(mask);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  CUDA_1D_KERNEL_LOOP(i, h2_n) {
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    y_h2[i] = __hmul2(__hmul2(x_h2[i], one_or_zero_h2), h2_scale);
  }
  if (n % 2 != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    const int64_t last_idx = n - 1;
    half one_or_zero = mask[last_idx];
    y[last_idx] = __hmul(__hmul(x[last_idx], one_or_zero), h2_scale.x);
  }
}

template<>
__global__ void MaskAndScaleAddGpu<half>(const int64_t n, float scale, const half* x,
                                         const int8_t* mask, const half* addend, half* y) {
  const int64_t h2_n = n / 2;
  half2 h2_scale = __float2half2_rn(scale);
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  const auto* addend_h2 = reinterpret_cast<const half2*>(addend);
  const auto* mask_c2 = reinterpret_cast<const char2*>(mask);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  CUDA_1D_KERNEL_LOOP(i, h2_n) {
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    y_h2[i] = __hadd2(__hmul2(__hmul2(x_h2[i], one_or_zero_h2), h2_scale), addend_h2[i]);
  }
  if (n % 2 != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    const int64_t last_idx = n - 1;
    half one_or_zero = mask[last_idx];
    y[last_idx] = __hadd(__hmul(__hmul(x[last_idx], one_or_zero), h2_scale.x), addend[last_idx]);
  }
}

template<typename T>
void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x, const int8_t* mask,
                  T* y) {
  MaskAndScaleGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, scale, x, mask, y);
}

template<>
void MaskAndScale<half>(DeviceCtx* ctx, const int64_t n, float scale, const half* x,
                        const int8_t* mask, half* y) {
  MaskAndScaleGpu<half>
      <<<BlocksNum4ThreadsNum(RoundUp(n, 2) / 2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, scale, x, mask, y);
}

template<typename T>
void MaskAndScaleAdd(DeviceCtx* ctx, const int64_t n, float scale, const T* x, const int8_t* mask,
                     const T* addend, T* y) {
  MaskAndScaleAddGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, scale, x, mask, addend, y);
}

template<>
void MaskAndScaleAdd<half>(DeviceCtx* ctx, const int64_t n, float scale, const half* x,
                           const int8_t* mask, const half* addend, half* y) {
  MaskAndScaleAddGpu<half>
      <<<BlocksNum4ThreadsNum(RoundUp(n, 2) / 2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, scale, x, mask, addend, y);
}

template<typename T>
class DropoutKernelGPU final : public user_op::OpKernel {
 public:
  DropoutKernelGPU() = default;
  ~DropoutKernelGPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const float scale = ctx->Attr<float>("scale");
    if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
      const user_op::Tensor* addend = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      MaskAndScaleAdd<T>(ctx->device_ctx(), in->shape().elem_cnt(), scale, in->dptr<T>(),
                         mask->dptr<int8_t>(), addend->dptr<T>(), out->mut_dptr<T>());
    } else {
      MaskAndScale<T>(ctx->device_ctx(), in->shape().elem_cnt(), scale, in->dptr<T>(),
                      mask->dptr<int8_t>(), out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_GPU(dtype)                                                \
  REGISTER_USER_KERNEL("dropout").SetCreateFn<DropoutKernelGPU<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                                  \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_DROPOUT_KERNEL_GPU(half)
REGISTER_DROPOUT_KERNEL_GPU(float)
REGISTER_DROPOUT_KERNEL_GPU(double)

template<typename T>
class DropoutGradKernelGPU final : public user_op::OpKernel {
 public:
  DropoutGradKernelGPU() = default;
  ~DropoutGradKernelGPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale = ctx->Attr<float>("scale");
    MaskAndScale<T>(ctx->device_ctx(), dy->shape().elem_cnt(), scale, dy->dptr<T>(),
                    mask->dptr<int8_t>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_GPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelGPU<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_GPU(half)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(double)

}  // namespace

}  // namespace oneflow
