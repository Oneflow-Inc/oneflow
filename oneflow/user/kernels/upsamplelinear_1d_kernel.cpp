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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

template<typename T>
class UpsampleLinear1DKernel final : public user_op::OpKernel {
 public:
  UpsampleLinear1DKernel() = default;
  ~UpsampleLinear1DKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float scale_factor = ctx->Attr<float>("scale_factor");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleLinearGrad1DKernel final : public user_op::OpKernel {
 public:
  UpsampleLinearGrad1DKernel() = default;
  ~UpsampleLinearGrad1DKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kCPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float scale_factor = ctx->Attr<float>("scale_factor");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLELINEAR1D_CPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("upsample_linear_1d")                                            \
      .SetCreateFn<UpsampleLinear1DKernel<dtype>>()                                     \
      .SetIsMatchedHob(                                                                 \
          (user_op::HobDeviceTag() == "cpu")                                            \
          & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));                 \
  REGISTER_USER_KERNEL("upsample_linear_1d_grad")                                        \
      .SetCreateFn<UpsampleLinearGrad1DKernel<dtype>>()                                 \
      .SetIsMatchedHob(                                                                 \
          (user_op::HobDeviceTag() == "cpu")                                            \
          & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLELINEAR1D_CPU_KERNEL(float)
REGISTER_UPSAMPLELINEAR1D_CPU_KERNEL(double)

}  // namespace oneflow
