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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/loss_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {

namespace {

using namespace loss;

template<typename T>
struct SmoothL1Functor {
  float beta_;
  float inv_beta_;
  T half_of_one_;
  SmoothL1Functor(float beta)
      : beta_(beta), inv_beta_(static_cast<float>(1.0 / beta)), half_of_one_(static_cast<T>(0.5)) {}

  __device__ __forceinline__ T operator()(T input_val, T target_val) const {
    const T abs_diff = abs(input_val - target_val);
    if (abs_diff < beta_) {
      return half_of_one_ * abs_diff * abs_diff * inv_beta_;
    } else {
      return abs_diff - half_of_one_ * beta_;
    }
  }
};

template<>
struct SmoothL1Functor<half> {
  half beta_;
  half inv_beta_;
  half zero_;
  half half_of_one_;
  SmoothL1Functor(float beta)
      : beta_(__float2half(beta)),
        inv_beta_(__float2half(static_cast<float>(1.0 / beta))),
        zero_(__float2half(0.f)),
        half_of_one_(__float2half(0.5f)) {}

  __device__ __forceinline__ half operator()(half input_val, half target_val) const {
    const half diff = input_val - target_val;
    const half abs_diff = diff < zero_ ? __hneg(diff) : diff;
    if (abs_diff < beta_) {
      return half_of_one_ * abs_diff * abs_diff * inv_beta_;
    } else {
      return abs_diff - half_of_one_ * beta_;
    }
  }
};

template<typename T>
struct SmoothL1GradFunctor {
  float beta_;
  float inv_beta_;
  T zero_;
  SmoothL1GradFunctor(float beta)
      : beta_(beta), inv_beta_(static_cast<float>(1.0 / beta)), zero_(GetZeroVal<T>()) {}

  __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val) const {
    const T diff = input_val - target_val;
    const T abs_diff = abs(diff);
    T dx_val;
    if (abs_diff < beta_) {
      dx_val = diff * inv_beta_;
    } else {
      dx_val = (diff > zero_) - (diff < zero_);
    }
    return dx_val * dy_val;
  }
};

template<>
struct SmoothL1GradFunctor<half> {
  half beta_;
  half inv_beta_;
  half zero_;
  half one_;
  SmoothL1GradFunctor(float beta)
      : beta_(__float2half(beta)),
        inv_beta_(__float2half(static_cast<float>(1.0 / beta))),
        zero_(__float2half(0.f)),
        one_(__float2half(1.f)) {}

  __device__ __forceinline__ half operator()(half input_val, half target_val, half dy_val) const {
    const half diff = input_val - target_val;
    const half abs_diff = diff < zero_ ? __hneg(diff) : diff;
    half dx_val;
    if (abs_diff < beta_) {
      dx_val = diff * inv_beta_;
    } else {
      dx_val = (diff > zero_) - (diff < zero_);
    }
    return dx_val * dy_val;
  }
};

template<typename T>
class SmoothL1LossKernel : public SimpleLossKernel<DeviceType::kCUDA, T, SmoothL1LossKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, T* out) const {
    const float beta = ctx->Attr<float>("beta");
    OF_CUDA_CHECK((cuda::elementwise::Binary(SmoothL1Functor<T>(beta), elem_cnt, out, input, target,
                                             ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }
};

template<typename T>
class SmoothL1LossGradKernel
    : public SimpleLossGradKernel<DeviceType::kCUDA, T, SmoothL1LossGradKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, const T* dy, T* dx) const {
    const float beta = ctx->Attr<float>("beta");
    OF_CUDA_CHECK(
        (cuda::elementwise::Ternary(SmoothL1GradFunctor<T>(beta), elem_cnt, dx, input, target, dy,
                                    ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }
};

}  // namespace

REGISTER_SIMPLE_LOSS_KERNEL_CUDA("smooth_l1_loss", SmoothL1LossKernel)
REGISTER_SIMPLE_LOSS_GRAD_KERNEL_CUDA("smooth_l1_loss_grad", SmoothL1LossGradKernel)

}  // namespace user_op
}  // namespace oneflow
