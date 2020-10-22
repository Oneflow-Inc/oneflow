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

namespace oneflow {

namespace {

template<DeviceType device_type>
class RepeatKernel final : public user_op::OpKernel {
 public:
  RepeatKernel() = default;
  ~RepeatKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt());
    CHECK_EQ(in->data_type(), out->data_type());
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<void>(), in->dptr<void>(),
                        in->shape().elem_cnt() * GetSizeOfDataType(in->data_type()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REPEAT_KERNEL(device)                                                \
  REGISTER_USER_KERNEL("repeat").SetCreateFn<RepeatKernel<device>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device));

OF_PP_FOR_EACH_TUPLE(REGISTER_REPEAT_KERNEL, DEVICE_TYPE_SEQ)

#undef REGISTER_REPEAT_KERNEL

}  // namespace

}  // namespace oneflow
