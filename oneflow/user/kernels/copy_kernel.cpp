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

class CopyKernel final : public user_op::OpKernel {
 public:
  CopyKernel() = default;
  ~CopyKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape_view();
    CHECK_EQ(out->shape_view(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);
    if (in_shape.elem_cnt() == 0) {
      // 0 shape tensor do not need copy
      return;
    } else {
      AutoMemcpy(ctx->stream(), out->mut_raw_dptr(), in->raw_dptr(),
                 in_shape.elem_cnt() * GetSizeOfDataType(in_data_type), out->mem_case(),
                 in->mem_case());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("copy").SetCreateFn<CopyKernel>();

}  // namespace
}  // namespace oneflow
