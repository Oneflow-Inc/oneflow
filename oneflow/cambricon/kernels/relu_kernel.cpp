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
#include <cstdint>
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluReluKernel final : public user_op::OpKernel {
 public:
  MluReluKernel() = default;
  ~MluReluKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);

    cnnlActivationMode_t mode = CNNL_ACTIVATION_RELU;
    cnnlNanPropagation_t relu_nan_opt = CNNL_NOT_PROPAGATE_NAN;
    float coef = 1.0;

    CnnlTensorDescriptor input_desc, output_desc;
    cnnlActivationDescriptor_t activation_desc = nullptr;
    input_desc.set(in);
    output_desc.set(out);

    OF_CNNL_CHECK(cnnlCreateActivationDescriptor(&activation_desc));
    OF_CNNL_CHECK(cnnlSetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef));
    OF_CNNL_CHECK(cnnlActivationForward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                        activation_desc, nullptr, input_desc.desc(), in->dptr(),
                                        nullptr, output_desc.desc(), out->mut_dptr()));
    OF_CNNL_CHECK(cnnlDestroyActivationDescriptor(activation_desc));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_MLU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("relu").SetCreateFn<MluReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_RELU_MLU_KERNEL(float)
REGISTER_RELU_MLU_KERNEL(float16)

}  // namespace oneflow
