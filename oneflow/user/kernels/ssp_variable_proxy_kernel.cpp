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
class SspVariableProxyKernel final : public user_op::OpKernel {
 public:
  SspVariableProxyKernel() = default;
  ~SspVariableProxyKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* var = ctx->Tensor4ArgNameAndIndex("var", 0);
    const user_op::Tensor* ref = ctx->Tensor4ArgNameAndIndex("ref", 0);
    CHECK_EQ(var->dptr(), ref->dptr());
    user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const ShapeView& in_shape = ref->shape();
    CHECK_EQ(value->shape(), in_shape);
    const DataType in_data_type = ref->data_type();
    CHECK_EQ(value->data_type(), in_data_type);
    Memcpy<device_type>(ctx->device_ctx(), value->mut_dptr<void>(), ref->dptr<void>(),
                        in_shape.elem_cnt() * GetSizeOfDataType(in_data_type));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SSP_VARIABLE_PROXY_KERNEL(device)                                              \
  REGISTER_USER_KERNEL("ssp_variable_proxy")                                                    \
      .SetCreateFn<SspVariableProxyKernel<device>>()                                            \
      .SetIsMatchedHob(user_op::HobDeviceTag() == device)                                       \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("ref", 0, "var", 0, true));                      \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_SSP_VARIABLE_PROXY_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_SSP_VARIABLE_PROXY_KERNEL(DeviceType::kGPU)
#endif

}  // namespace

}  // namespace oneflow
