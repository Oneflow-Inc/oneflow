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
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class AccKernel final : public user_op::OpKernel {
 public:
  AccKernel() = default;
  ~AccKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt());
    CHECK_EQ(in->data_type(), out->data_type());
    KernelUtil<device_type, T>::Axpy(ctx->device_ctx(), in->shape().elem_cnt(), GetOneVal<T>(),
                                     in->dptr<T>(), 1, out->mut_dptr<T>(), 1);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ACC_KERNEL(device, dtype)                       \
  REGISTER_USER_KERNEL("acc")                                    \
      .SetCreateFn<AccKernel<device, OF_PP_PAIR_FIRST(dtype)>>() \
      .SetIsMatchedHob(                                          \
          (user_op::HobDeviceTag() == device)                    \
          & (user_op::HobDataType("out", 0) == GetDataType<OF_PP_PAIR_FIRST(dtype)>::value));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ACC_KERNEL, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

#undef REGISTER_ACC_KERNEL

}  // namespace

}  // namespace oneflow
