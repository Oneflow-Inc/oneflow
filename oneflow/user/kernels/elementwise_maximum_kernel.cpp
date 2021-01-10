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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace user_op {

template<typename T>
class CpuElementwiseMaximumKernel final : public user_op::OpKernel {
 public:
  CpuElementwiseMaximumKernel() = default;
  ~CpuElementwiseMaximumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    const T* x = tensor_x->dptr<T>();
    const T* y = tensor_y->dptr<T>();
    T* z = tensor_z->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    FOR_RANGE(int64_t, i, 0, n) { z[i] = x[i] > y[i] ? x[i] : y[i]; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class CpuElementwiseMaximumBackwardKernel final : public user_op::OpKernel {
 public:
  CpuElementwiseMaximumBackwardKernel() = default;
  ~CpuElementwiseMaximumBackwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const T* dptr_dz = tensor_dz->dptr<T>();
    const T* dptr_x = tensor_x->dptr<T>();
    const T* dptr_y = tensor_y->dptr<T>();

    T* dptr_dx = tensor_dx ? tensor_dx->mut_dptr<T>() : nullptr;
    T* dptr_dy = tensor_dx ? tensor_dy->mut_dptr<T>() : nullptr;

    FOR_RANGE(int64_t, idx, 0, tensor_dz->shape().elem_cnt()) {
      if (dptr_x[idx] > dptr_y[idx]) {
        if (dptr_dx) { dptr_dx[idx] = dptr_dz[idx]; }
      } else {
        if (dptr_dy) { dptr_dy[idx] = dptr_dz[idx]; }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MAXIMUM_CPU_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("elementwise_maximum")                                        \
      .SetCreateFn<CpuElementwiseMaximumKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)                 \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MAXIMUM_CPU_KERNEL(float);
REGISTER_MAXIMUM_CPU_KERNEL(double);

#define REGISTER_BW_MAXIMUM_CPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("elementwise_maximum_backward")                             \
      .SetCreateFn<CpuElementwiseMaximumBackwardKernel<DeviceType::kCPU, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)               \
                       & (user_op::HobDataType("dz", 0) == GetDataType<dtype>::value));

REGISTER_BW_MAXIMUM_CPU_KERNEL(float);
REGISTER_BW_MAXIMUM_CPU_KERNEL(double);
}  // namespace user_op
}  // namespace oneflow
