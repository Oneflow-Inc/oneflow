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
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {

namespace {

using namespace loss;

template<typename T>
__global__ void ComputeSmoothL1Out(int64_t elem_cnt, const T* input, const T* target, T* out,
                                   const float beta, const float inv_beta) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T abs_diff = abs(input[i] - target[i]);
    if (abs_diff < beta) {
      out[i] = 0.5 * abs_diff * abs_diff * inv_beta;
    } else {
      out[i] = abs_diff - 0.5 * beta;
    }
  }
}

template<>
__global__ void ComputeSmoothL1Out(int64_t elem_cnt, const half* input, const half* target,
                                   half* out, const float beta, const float inv_beta) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const half half_zero = __float2half(0.0);
  const half half_one = __float2half(0.5);
  const half half_beta = __float2half(beta);
  const half half_inv_beta = __float2half(inv_beta);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half diff = __hsub(input[i], target[i]);
    const half abs_diff = __hlt(diff, half_zero) ? __hneg(diff) : diff;
    if (__hlt(abs_diff, half_beta)) {
      out[i] = __hmul(__hmul(half_one, abs_diff), __hmul(abs_diff, half_inv_beta));
    } else {
      out[i] = __hsub(abs_diff, __hmul(half_one, half_beta));
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
__global__ void ComputeSmoothL1GradOut(int64_t elem_cnt, float inv_elem_cnt, const T* input,
                                       const T* target, const T* dy, T* dx,
                                       const ReductionType reduction_type, const float beta,
                                       const float inv_beta) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T diff = input[i] - target[i];
    const T abs_diff = abs(diff);
    T dx_val;
    if (abs_diff < beta) {
      dx_val = diff * inv_beta;
    } else {
      dx_val = (diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>());
    }
    const T dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
    dx_val = dx_val * dy_val;
    if (reduction_type == ReductionType::kMean) { dx_val *= inv_elem_cnt; };
    dx[i] = dx_val;
  }
}

template<>
__global__ void ComputeSmoothL1GradOut(int64_t elem_cnt, float inv_elem_cnt, const half* input,
                                       const half* target, const half* dy, half* dx,
                                       const ReductionType reduction_type, const float beta,
                                       const float inv_beta) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const half half_zero = __float2half(0.0);
  const half half_one = __float2half(1.0);
  const half half_beta = __float2half(beta);
  const half half_inv_beta = __float2half(inv_beta);
  const half half_inv_elem_cnt = __float2half(inv_elem_cnt);

  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half diff = __hsub(input[i], target[i]);
    const half abs_diff = __hlt(diff, half_zero) ? __hneg(diff) : diff;
    half dx_val;
    if (__hlt(abs_diff, half_beta)) {
      dx_val = __hmul(diff, half_inv_beta);
    } else {
      const half left = __hgt(diff, half_zero) ? half_one : half_zero;
      const half right = __hlt(diff, half_zero) ? half_one : half_zero;
      dx_val = __hsub(left, right);
    }
    const half dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
    dx_val = __hmul(dx_val, dy_val);
    if (reduction_type == ReductionType::kMean) { dx_val = __hmul(dx_val, half_inv_elem_cnt); };
    dx[i] = dx_val;
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
class SmoothL1LossKernel : public SimpleLossKernel<DeviceType::kCUDA, T, SmoothL1LossKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, T* out) const {
    const float beta = ctx->Attr<float>("beta");
    ComputeSmoothL1Out<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, input, target, out, beta, static_cast<float>(1.0 / beta));
  }
};

template<typename T>
class SmoothL1LossGradKernel
    : public SimpleLossGradKernel<DeviceType::kCUDA, T, SmoothL1LossGradKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, const T* dy, T* dx, const ReductionType reduction) const {
    const float beta = ctx->Attr<float>("beta");
    ComputeSmoothL1GradOut<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, static_cast<float>(1.0 / elem_cnt), input, target, dy, dx, reduction, beta,
        static_cast<float>(1.0 / beta));
  }
};

}  // namespace

REGISTER_SIMPLE_LOSS_KERNEL_CUDA("smooth_l1_loss", SmoothL1LossKernel)
REGISTER_SIMPLE_LOSS_GRAD_KERNEL_CUDA("smooth_l1_loss_grad", SmoothL1LossGradKernel)

}  // namespace user_op
}  // namespace oneflow
