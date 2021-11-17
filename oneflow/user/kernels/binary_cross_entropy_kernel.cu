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
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T>
__global__ void ComputeBinaryCrossEntropyOut(int64_t elem_cnt, const T* input, const T* target,
                                             T* out, const T* weight) {
  const T zero_val = GetZeroVal<T>();
  const T one_val = GetOneVal<T>();
  const T negative_hundred_val = -100 * one_val;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    assert(input_val >= zero_val);
    assert(input_val <= one_val);
    T out_val =
        (target_val - one_val) * max(static_cast<T>(log(one_val - input_val)), negative_hundred_val)
        - target_val * max(static_cast<T>(log(input_val)), negative_hundred_val);
    if (weight != nullptr) { out_val *= weight[i]; }
    out[i] = out_val;
  }
}

template<>
__global__ void ComputeBinaryCrossEntropyOut(int64_t elem_cnt, const half* input,
                                             const half* target, half* out, const half* weight) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const float zero_val = 0.0;
  const float one_val = 1.0;
  const float negative_hundred_val = -100;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    float input_val = __half2float(input[i]);
    float target_val = __half2float(target[i]);

    assert(input_val >= zero_val);
    assert(input_val <= one_val);

    half out_val =
        __float2half((target_val - one_val) * max(__logf(one_val - input_val), negative_hundred_val)
                     - target_val * max(__logf(input_val), negative_hundred_val));
    if (weight != nullptr) { out_val = __hmul(out_val, weight[i]); }
    out[i] = out_val;
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
__global__ void ComputeBinaryCrossEntropyGradOut(int64_t elem_cnt, float inv_elem_cnt,
                                                 const T* input, const T* target, const T* dy,
                                                 T* dx, const T* weight,
                                                 const ReductionType reduction_type) {
  const T eps = static_cast<T>(1e-12);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    T dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
    T dx_val =
        dy_val * (input_val - target_val) / max((static_cast<T>(1.0) - input_val) * input_val, eps);
    if (weight != nullptr) { dx_val *= weight[i]; }
    if (reduction_type == ReductionType::kMean) { dx_val *= inv_elem_cnt; }
    dx[i] = dx_val;
  }
}

template<>
__global__ void ComputeBinaryCrossEntropyGradOut(int64_t elem_cnt, float inv_elem_cnt,
                                                 const half* input, const half* target,
                                                 const half* dy, half* dx, const half* weight,
                                                 const ReductionType reduction_type) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const float one_val = 1.0;
  const float eps = 1e-12;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    float input_val = __half2float(input[i]);
    float target_val = __half2float(target[i]);
    float dy_val = __half2float(reduction_type == ReductionType::kNone ? dy[i] : *dy);
    half dx_val = __float2half(dy_val * (input_val - target_val)
                               / max((one_val - input_val) * input_val, eps));
    if (weight != nullptr) { dx_val = __hmul(dx_val, weight[i]); }
    if (reduction_type == ReductionType::kMean) {
      dx_val = __float2half(__half2float(dx_val) * inv_elem_cnt);
    }
    dx[i] = dx_val;
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
class BinaryCrossEntropyKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyKernel() = default;
  ~BinaryCrossEntropyKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();
    T* tmp_out = reduction == ReductionType::kNone ? nullptr : tmp_buffer_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;
    ComputeBinaryCrossEntropyOut<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                   ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, input, target, reduction == ReductionType::kNone ? out : tmp_out, weight);

    ApplyLossReductionIfNeed<DeviceType::kGPU, T>(ctx->device_ctx(), elem_cnt, tmp_out, out,
                                                  reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class BinaryCrossEntropyGradKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyGradKernel() = default;
  ~BinaryCrossEntropyGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;
    ComputeBinaryCrossEntropyGradOut<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                       ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, static_cast<float>(1.0 / elem_cnt), input, target, dy, dx, weight, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("binary_cross_entropy")                                            \
      .SetCreateFn<BinaryCrossEntropyKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                     \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn(loss::GenDefaultInferTmpSizeFn<dtype>());

#define REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(dtype)                                  \
  REGISTER_USER_KERNEL("binary_cross_entropy_grad")                                       \
      .SetCreateFn<BinaryCrossEntropyGradKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                     \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_BINARY_CROSS_ENTROPY_KERNEL(half)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double)

REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(half)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
