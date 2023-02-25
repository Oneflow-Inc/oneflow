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

class CopyHdKernel final : public user_op::OpKernel {
 public:
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK(in) << "input of copy not found";
    const ShapeView& in_shape = in->shape_view();
    if (in_shape.elem_cnt() == 0) {
      // 0 shape tensor do not need copy
    } else {
      const DataType in_data_type = in->data_type();
      user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
      CHECK(out) << "output of copy not found, op: " << ctx->op_name();
      CHECK_EQ(out->shape_view(), in_shape);
      CHECK_EQ(out->data_type(), in_data_type);

      ep::primitive::MemcpyKind kind{};
      if (ctx->op_type_name() == "copy_h2d") {
        kind = ep::primitive::MemcpyKind::kHtoD;
      } else if (ctx->op_type_name() == "copy_d2h") {
        kind = ep::primitive::MemcpyKind::kDtoH;
      } else {
        UNIMPLEMENTED();
      }
      std::unique_ptr<ep::primitive::Memcpy> primitive =
          ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->stream()->device_type(),
                                                                    kind);
      primitive->Launch(ctx->stream(), out->mut_raw_dptr(), in->raw_dptr(),
                        in_shape.elem_cnt() * GetSizeOfDataType(in_data_type));
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("copy_h2d").SetCreateFn<CopyHdKernel>();
REGISTER_USER_KERNEL("copy_d2h").SetCreateFn<CopyHdKernel>();

}  // namespace
}  // namespace oneflow
