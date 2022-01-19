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
void ComputeSmoothL1Out(int64_t elem_cnt, const T* input, const T* target, T* out,
                        const float beta) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    const T abs_diff = std::abs(input[i] - target[i]);
    if (abs_diff < beta) {
      out[i] = 0.5 * abs_diff * abs_diff / beta;
    } else {
      out[i] = abs_diff - 0.5 * beta;
    }
  }
}
template<typename T>
void ComputeSmoothL1GradOut(int64_t elem_cnt, const T* input, const T* target, const T* dy, T* dx,
                            const float beta) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    const T diff = input[i] - target[i];
    const T abs_diff = std::abs(diff);
    if (abs_diff < beta) {
      dx[i] = diff / beta;
    } else {
      dx[i] = (diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>());
    }
    const T dy_val = dy[i];
    dx[i] = dx[i] * dy_val;
  }
}

template<typename T>
class SmoothL1LossKernel : public SimpleLossKernel<DeviceType::kCPU, T, SmoothL1LossKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, T* out) const {
    const float beta = ctx->Attr<float>("beta");
    ComputeSmoothL1Out(elem_cnt, input, target, out, beta);
  }
};

template<typename T>
class SmoothL1LossGradKernel
    : public SimpleLossGradKernel<DeviceType::kCPU, T, SmoothL1LossGradKernel<T>> {
 public:
  void ComputeOut(user_op::KernelComputeContext* ctx, int64_t elem_cnt, const T* input,
                  const T* target, const T* dy, T* dx) const {
    const float beta = ctx->Attr<float>("beta");
    ComputeSmoothL1GradOut(elem_cnt, input, target, dy, dx, beta);
  }
};

}  // namespace

REGISTER_SIMPLE_LOSS_KERNEL_CPU("smooth_l1_loss", SmoothL1LossKernel)
REGISTER_SIMPLE_LOSS_GRAD_KERNEL_CPU("smooth_l1_loss_grad", SmoothL1LossGradKernel)

}  // namespace user_op
}  // namespace oneflow
