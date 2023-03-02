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

class CastToStaticShapeKernel final : public user_op::OpKernel {
 public:
  CastToStaticShapeKernel() = default;
  ~CastToStaticShapeKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Shape& input_static_shape = ctx->TensorDesc4ArgNameAndIndex("input", 0)->shape();
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    CHECK(input_tensor->shape_view() == ShapeView(input_static_shape));
    CHECK_EQ(output_tensor->shape_view(), input_tensor->shape_view());
    size_t output_tensor_size =
        output_tensor->shape_view().elem_cnt() * GetSizeOfDataType(output_tensor->data_type());
    std::unique_ptr<ep::primitive::Memcpy> primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->stream()->device_type(),
                                                                  ep::primitive::MemcpyKind::kDtoD);
    CHECK(primitive) << "Can not create Memcpy primitive for device type "
                     << ctx->stream()->device_type();
    primitive->Launch(ctx->stream(), output_tensor->mut_dptr(), input_tensor->dptr(),
                      output_tensor_size);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

REGISTER_USER_KERNEL("cast_to_static_shape")
    .SetCreateFn<CastToStaticShapeKernel>()
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("output", 0, "input", 0, false));
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
