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
void ComputeKLDivOut(int64_t elem_cnt, const T* input, const T* target, T* out,
                     const bool log_target) {
  if (log_target) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = std::exp(target[i]) * (target[i] - input[i]); }
  } else {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      const auto out_val = target[i] * (SafeLog(target[i]) - input[i]);
      out[i] = target[i] > 0 ? out_val : static_cast<T>(0);
    }
  }
}

template<typename T>
void ComputeKLDivGradOut(int64_t elem_cnt, const T* input, const T* target, const T* dy, T* dx,
                         const bool log_target) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    const T dy_val = dy[i];
    dx[i] =
        log_target ? (-std::exp(target[i]) * dy_val) : (target[i] > 0 ? -target[i] * dy_val : 0);
  }
}

template<typename T>
class KLDivKernel : public SimpleLossKernel<DeviceType::kCPU, T, KLDivKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, T* out) const {
    const bool log_target = ctx->Attr<bool>("log_target");
    ComputeKLDivOut(elem_cnt, input, target, out, log_target);
  }
};

template<typename T>
class KLDivGradKernel : public SimpleLossGradKernel<DeviceType::kCPU, T, KLDivGradKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, const T* dy, T* dx) const {
    const bool log_target = ctx->Attr<bool>("log_target");
    ComputeKLDivGradOut(elem_cnt, input, target, dy, dx, log_target);
  }
};

}  // namespace

REGISTER_SIMPLE_LOSS_KERNEL_CPU("kl_div_loss", KLDivKernel)
REGISTER_SIMPLE_LOSS_GRAD_KERNEL_CPU("kl_div_loss_grad", KLDivGradKernel)

}  // namespace user_op
}  // namespace oneflow
