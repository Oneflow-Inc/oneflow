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
#include "oneflow/user/kernels/in_top_k_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class InTopkKernel final : public user_op::OpKernel {
 public:
  InTopkKernel() = default;
  ~InTopkKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* targets = ctx->Tensor4ArgNameAndIndex("targets", 0);
    const user_op::Tensor* predictions = ctx->Tensor4ArgNameAndIndex("predictions", 0);
    const int32_t k = ctx->Attr<int32_t>("k");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(targets->shape().At(0), predictions->shape().At(0));
    CHECK_EQ(targets->shape().NumAxes(), 1);
    CHECK_EQ(predictions->shape().NumAxes(), 2);
    const int32_t instance_num = predictions->shape().At(0);
    const int32_t classes_num = predictions->shape().At(1);
    InTopkKernelUtil<device_type, T>::InTopk(ctx->device_ctx(), instance_num, classes_num,
                                             targets->dptr<T>(), predictions->dptr<float>(), k,
                                             out->mut_dptr<int8_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_IN_TOP_K_KERNEL(device, target_dtype_pair)                     \
  REGISTER_USER_KERNEL("in_top_k")                                              \
      .SetCreateFn<InTopkKernel<device, OF_PP_PAIR_FIRST(target_dtype_pair)>>() \
      .SetIsMatchedHob(                                                         \
          (user_op::HobDeviceTag() == device)                                   \
          & (user_op::HobDataType("targets", 0) == OF_PP_PAIR_SECOND(target_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_IN_TOP_K_KERNEL, DEVICE_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#undef REGISTER_IN_TOP_K_KERNEL

}  // namespace oneflow
