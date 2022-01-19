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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/loss_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T, bool LOG_TARGET>
struct KLDivFunctor {
  __device__ __forceinline__ T operator()(T input_val, T target_val) const {
    if (LOG_TARGET) {
      return exp(target_val) * (target_val - input_val);
    } else {
      const T zero_val = static_cast<T>(0);
      const T out_val = target_val * (SafeLog(target_val) - input_val);
      return target_val > zero_val ? out_val : zero_val;
    }
  }
};

template<bool LOG_TARGET>
struct KLDivFunctor<half, LOG_TARGET> {
  __device__ __forceinline__ half operator()(half input_val, half target_val) const {
    if (LOG_TARGET) {
      return hexp(target_val) * (target_val - input_val);
    } else {
      const half zero_val = __float2half(0.f);
      const half out_val = target_val * (SafeLog(target_val) - input_val);
      return target_val > zero_val ? out_val : zero_val;
    }
  }
};

template<typename T, bool LOG_TARGET>
struct KLDivGradFunctor {
  __device__ __forceinline__ T operator()(T target_val, T dy_val) const {
    if (LOG_TARGET) {
      return -exp(target_val) * dy_val;
    } else {
      const T zero_val = static_cast<T>(0);
      return target_val > zero_val ? -target_val * dy_val : zero_val;
    }
  }
};

template<bool LOG_TARGET>
struct KLDivGradFunctor<half, LOG_TARGET> {
  __device__ __forceinline__ half operator()(half target_val, half dy_val) const {
    if (LOG_TARGET) {
      return __hneg(hexp(target_val) * dy_val);
    } else {
      const half zero_val = __float2half(0.f);
      return target_val > zero_val ? __hneg(target_val * dy_val) : zero_val;
    }
  }
};

template<typename T>
class KLDivKernel : public SimpleLossKernel<DeviceType::kCUDA, T, KLDivKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, T* out) const {
    const bool log_target = ctx->Attr<bool>("log_target");
    if (log_target) {
      OF_CUDA_CHECK(
          (cuda::elementwise::Binary(KLDivFunctor<T, true>(), elem_cnt, out, input, target,
                                     ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    } else {
      OF_CUDA_CHECK(
          (cuda::elementwise::Binary(KLDivFunctor<T, false>(), elem_cnt, out, input, target,
                                     ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    }
  }
};

template<typename T>
class KLDivGradKernel : public SimpleLossGradKernel<DeviceType::kCUDA, T, KLDivGradKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, const T* dy, T* dx) const {
    const bool log_target = ctx->Attr<bool>("log_target");
    if (log_target) {
      OF_CUDA_CHECK((cuda::elementwise::Binary(
          KLDivGradFunctor<T, /*LOG_TARGET*/ true>(), elem_cnt, dx, target, dy,
          ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    } else {
      OF_CUDA_CHECK((cuda::elementwise::Binary(
          KLDivGradFunctor<T, /*LOG_TARGET*/ false>(), elem_cnt, dx, target, dy,
          ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    }
  }
};

}  // namespace

REGISTER_SIMPLE_LOSS_KERNEL_CUDA("kl_div_loss", KLDivKernel)
REGISTER_SIMPLE_LOSS_GRAD_KERNEL_CUDA("kl_div_loss_grad", KLDivGradKernel)

}  // namespace user_op
}  // namespace oneflow
