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
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace user_op {

class EmptyKernel final : public OpKernel {
 public:
  EmptyKernel() = default;
  ~EmptyKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    // None POD type need check
    if (!IsTriviallyCopyableDataType(out->data_type())) {
      CHECK(out->shape_view().NumAxes() > 0 && out->shape_view().elem_cnt() == 0)
          << "None POD Tensor created by empty op must be 0-Size tensor.";
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("empty").SetCreateFn<EmptyKernel>();

}  // namespace user_op
}  // namespace oneflow
