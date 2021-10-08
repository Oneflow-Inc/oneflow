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
  if (log_target) {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out[i] = exp(target[i]) * (target[i] - input[i]); }
  } else {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      const auto out_val = target[i] * (SafeLog(target[i]) - input[i]);
      out[i] = target[i] > 0 ? out_val : static_cast<T>(0);
    }
  }
}

template<>
__global__ void ComputeKLDivOut(int64_t elem_cnt, const float16* input, const float16* target,
                                float16* out, const bool log_target) {
  FLOAT16_TO_HALF(input)
  FLOAT16_TO_HALF(target)
  FLOAT16_TO_HALF(out)

  const half zero_half = __float2half(0.0);
  if (log_target) {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      out_[i] = __hmul(hexp(target_[i]), __hsub(target_[i], input_[i]));
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      const half out_val = __hmul(target_[i], __hsub(SafeLog(target_[i]), input_[i]));
      out_[i] = __hgt(target_[i], zero_half) ? out_val : zero_half;
    }
  }
}

template<typename T>
__global__ void ComputeKLDivGradOut(int64_t elem_cnt, const T* input, const T* target, const T* dy,
                                    T* dx, const ReductionType reduction_type,
                                    const bool log_target) {
#define SET_DY_VAL const T dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
#define DEAL_REDUCE_MEAN \
  if (reduction_type == ReductionType::kMean) { dx[i] /= elem_cnt; }

  {
    if (log_target) {
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        SET_DY_VAL
        dx[i] = -exp(target[i]) * dy_val;
        DEAL_REDUCE_MEAN
      }
    } else {
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        SET_DY_VAL
        dx[i] = target[i] > 0 ? -target[i] * dy_val : 0;
        DEAL_REDUCE_MEAN
      }
    }
  }

#undef SET_DY_VAL
#undef DEAL_REDUCE_MEAN
}

template<>
__global__ void ComputeKLDivGradOut(int64_t elem_cnt, const float16* input, const float16* target,
                                    const float16* dy, float16* dx,
                                    const ReductionType reduction_type, const bool log_target) {
  FLOAT16_TO_HALF(input)
  FLOAT16_TO_HALF(target)
  FLOAT16_TO_HALF(dy)
  FLOAT16_TO_HALF(dx)
#define SET_DY_VAL const half dy_val = reduction_type == ReductionType::kNone ? dy_[i] : *dy_;
#define DEAL_REDUCE_MEAN                                                 \
  if (reduction_type == ReductionType::kMean) {                          \
    dx_[i] = __hdiv(dx_[i], __float2half(static_cast<float>(elem_cnt))); \
  }

  {
    const half zero_half = __float2half(0.0);
    if (log_target) {
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        SET_DY_VAL
        dx_[i] = __hneg(__hmul(hexp(target_[i]), dy_val));
        DEAL_REDUCE_MEAN
      }
    } else {
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        SET_DY_VAL
        dx_[i] = __hgt(target_[i], zero_half) ? __hneg(__hmul(target_[i], dy_val)) : zero_half;
        DEAL_REDUCE_MEAN
      }
    }
  }

#undef SET_DY_VAL
#undef DEAL_REDUCE_MEAN
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
                          ctx->device_ctx()->cuda_stream()>>>(elem_cnt, input, target, dy, dx,
                                                              reduction, log_target);
  }
};

}  // namespace

REGISTER_SIMPLE_LOSS_KERNEL_GPU("kl_div_loss", KLDivKernel)
REGISTER_SIMPLE_LOSS_GRAD_KERNEL_GPU("kl_div_loss_grad", KLDivGradKernel)

}  // namespace user_op
}  // namespace oneflow
