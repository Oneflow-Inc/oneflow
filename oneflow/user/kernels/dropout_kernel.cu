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

template<>
__global__ void MaskAndScaleGpu<half>(const int64_t n, float scale, const half* x,
                                      const int8_t* mask, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half h_scale = __float2half(scale);
  CUDA_1D_KERNEL_LOOP(i, n) {
    half one_or_zero = mask[i];
    y[i] = __hmul(__hmul(x[i], one_or_zero), h_scale);
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

__global__ void GenMaskGpu(const int64_t n, float threshold, const float* random_tmp,
                           int8_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, n) { mask[i] = random_tmp[i] > threshold; }
}

template<typename T>
void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x, const int8_t* mask,
                  T* y) {
  MaskAndScaleGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, scale, x, mask, y);
}

template<>
void MaskAndScale<float16>(DeviceCtx* ctx, const int64_t n, float scale, const float16* x,
                           const int8_t* mask, float16* y) {
  MaskAndScaleGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, scale, reinterpret_cast<const half*>(x), mask, reinterpret_cast<half*>(y));
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
    MaskAndScale<T>(ctx->device_ctx(), in->shape().elem_cnt(), scale, in->dptr<T>(),
                    mask->dptr<int8_t>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_GPU(dtype)                                                      \
  REGISTER_USER_KERNEL("dropout")                                                               \
      .SetCreateFn<DropoutKernelGPU<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_KERNEL_GPU(float16)
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

REGISTER_DROPOUT_GRAD_KERNEL_GPU(float16)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(double)

}  // namespace

}  // namespace oneflow
