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
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluClipByScalarMaxKernel final : public user_op::OpKernel {
 public:
  MluClipByScalarMaxKernel() = default;
  ~MluClipByScalarMaxKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    float floating_max = static_cast<float>(ctx->Attr<double>("floating_max"));
    int32_t integral_max = static_cast<int32_t>(ctx->Attr<int64_t>("integral_max"));
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(x);
    output_desc.set(y);
    void* max = nullptr;
    if (x->data_type() == DataType::kInt32) {
      max = &integral_max;
    } else {
      max = &floating_max;
    }
    OF_CNNL_CHECK(cnnlClip_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                              CNNL_POINTER_MODE_HOST, input_desc.desc(), x->dptr(), nullptr, max,
                              output_desc.desc(), y->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CLIP_BY_VALUE_MLU_KERNEL(op_type_name, kernel_name, dtype) \
  REGISTER_USER_KERNEL(op_type_name)                                        \
      .SetCreateFn<kernel_name##Kernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)       \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CLIP_BY_VALUE_MLU_KERNEL("clip_by_scalar_max", MluClipByScalarMax, float)
REGISTER_CLIP_BY_VALUE_MLU_KERNEL("clip_by_scalar_max", MluClipByScalarMax, float16)
REGISTER_CLIP_BY_VALUE_MLU_KERNEL("clip_by_scalar_max", MluClipByScalarMax, int32_t)

#undef REGISTER_CLIP_BY_VALUE_MLU_KERNEL

}  // namespace oneflow
