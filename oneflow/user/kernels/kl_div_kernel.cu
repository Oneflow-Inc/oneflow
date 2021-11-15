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
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T>
__global__ void ComputeKLDivOut(int64_t elem_cnt, const T* input, const T* target, T* out,
                                const bool log_target) {
  const T zero_val = static_cast<T>(0);
  if (log_target) {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      const T target_val = target[i];
      out[i] = exp(target_val) * (target_val - input[i]);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      const T target_val = target[i];
      const auto out_val = target_val * (SafeLog(target_val) - input[i]);
      out[i] = target_val > zero_val ? out_val : zero_val;
    }
  }
}

template<>
__global__ void ComputeKLDivOut(int64_t elem_cnt, const half* input, const half* target, half* out,
                                const bool log_target) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const half zero_val = __float2half(0.0);
  if (log_target) {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      const half target_val = target[i];
      out[i] = __hmul(hexp(target_val), __hsub(target_val, input[i]));
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      const half target_val = target[i];
      const half out_val = __hmul(target_val, __hsub(SafeLog(target_val), input[i]));
      out[i] = __hgt(target_val, zero_val) ? out_val : zero_val;
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
__global__ void ComputeKLDivGradOut(int64_t elem_cnt, float inv_elem_cnt, const T* input,
                                    const T* target, const T* dy, T* dx,
                                    const ReductionType reduction_type, const bool log_target) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T target_val = target[i];
    const T dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
    T dx_val;
    dx_val = log_target ? -exp(target_val) * dy_val : target_val > 0 ? -target_val * dy_val : 0;
    if (reduction_type == ReductionType::kMean) { dx_val *= inv_elem_cnt; }
    dx[i] = dx_val;
  }
}

template<>
__global__ void ComputeKLDivGradOut(int64_t elem_cnt, float inv_elem_cnt, const half* input,
                                    const half* target, const half* dy, half* dx,
                                    const ReductionType reduction_type, const bool log_target) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const half zero_val = __float2half(0.0);
  const half half_inv_elem_cnt = __float2half(inv_elem_cnt);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half target_val = target[i];
    const half dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
    half dx_val;
    dx_val = log_target
                 ? __hneg(__hmul(hexp(target_val), dy_val))
                 : (__hgt(target_val, zero_val) ? __hneg(__hmul(target_val, dy_val)) : zero_val);
    if (reduction_type == ReductionType::kMean) { dx_val = __hmul(dx_val, half_inv_elem_cnt); }
    dx[i] = dx_val;
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
class KLDivKernel : public SimpleLossKernel<DeviceType::kGPU, T, KLDivKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, T* out) const {
    const bool log_target = ctx->Attr<bool>("log_target");
    ComputeKLDivOut<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                      ctx->device_ctx()->cuda_stream()>>>(elem_cnt, input, target, out, log_target);
  }
};

template<typename T>
class KLDivGradKernel : public SimpleLossGradKernel<DeviceType::kGPU, T, KLDivGradKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, const T* dy, T* dx, const ReductionType reduction) const {
    const bool log_target = ctx->Attr<bool>("log_target");
    ComputeKLDivGradOut<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, static_cast<float>(1.0 / elem_cnt), input, target, dy, dx, reduction, log_target);
  }
};

}  // namespace

REGISTER_SIMPLE_LOSS_KERNEL_GPU("kl_div_loss", KLDivKernel)
REGISTER_SIMPLE_LOSS_GRAD_KERNEL_GPU("kl_div_loss_grad", KLDivGradKernel)

}  // namespace user_op
}  // namespace oneflow
