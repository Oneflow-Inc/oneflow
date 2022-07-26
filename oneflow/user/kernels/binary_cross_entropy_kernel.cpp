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
void ComputeBinaryCrossEntropyOut(int64_t elem_cnt, const T* input, const T* target, T* out,
                                  const T* weight) {
  T negative_100 = static_cast<T>(-100);
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    CHECK_LE(input_val, 1.0);
    CHECK_GE(input_val, 0.0);
    out[i] = (target_val - 1) * std::max(static_cast<T>(std::log(1.0 - input_val)), negative_100)
             - target_val * std::max(static_cast<T>(std::log(input_val)), negative_100);
    if (weight != nullptr) { out[i] *= weight[i]; }
  }
}

template<typename T>
void ComputeBinaryCrossEntropyGradOut(int64_t elem_cnt, const T* input, const T* target,
                                      const T* dy, T* dx, const T* weight) {
  const T eps = static_cast<T>(1e-12);
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    T dy_val = dy[i];
    dx[i] = dy_val * (input_val - target_val)
            / (std::max((static_cast<T>(1.0) - input_val) * input_val, eps));
    if (weight != nullptr) { dx[i] *= weight[i]; }
  }
}
template<typename T>
class BinaryCrossEntropyKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyKernel() = default;
  ~BinaryCrossEntropyKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;

    ComputeBinaryCrossEntropyOut(elem_cnt, input, target, out, weight);
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

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;
    ComputeBinaryCrossEntropyGradOut(elem_cnt, input, target, dy, dx, weight);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("binary_cross_entropy")                                             \
      .SetCreateFn<BinaryCrossEntropyKernel<dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("binary_cross_entropy_grad")                                        \
      .SetCreateFn<BinaryCrossEntropyGradKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
