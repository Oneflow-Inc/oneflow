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
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluDivKernel final : public user_op::OpKernel {
 public:
  MluDivKernel() = default;
  ~MluDivKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);

    CnnlTensorDescriptor x_desc, y_decs, z_desc;
    x_desc.set(x);
    y_decs.set(y);
    z_desc.set(z);
    size_t _div_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetDivWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                          x_desc.desc(), y_decs.desc(), z_desc.desc(),
                                          &_div_workspace_size));
    CnnlWorkspace cnnl_workspace(ctx->stream()->As<ep::MluStream>(), _div_workspace_size);
    void* _div_workspace = cnnl_workspace.dptr();

    OF_CNNL_CHECK(cnnlDiv_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                             CNNL_COMPUTATION_HIGH_PRECISION, x_desc.desc(), x->dptr(),
                             y_decs.desc(), y->dptr(), _div_workspace, _div_workspace_size,
                             z_desc.desc(), z->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIV_MLU_KERNEL(name, dtype)                                     \
  REGISTER_USER_KERNEL(name).SetCreateFn<MluDivKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                             \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_DIV_MLU_KERNEL("div", float)
REGISTER_DIV_MLU_KERNEL("div", float16)
REGISTER_DIV_MLU_KERNEL("broadcast_div", float)
REGISTER_DIV_MLU_KERNEL("broadcast_div", float16)

}  // namespace oneflow
