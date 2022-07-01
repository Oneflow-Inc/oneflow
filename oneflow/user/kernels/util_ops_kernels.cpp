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
#include "oneflow/user/kernels/elementwise_primitive_kernel.h"

namespace oneflow {
namespace user_op {

REGISTER_USER_KERNEL("isinf")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kIsInf, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kIsInf, "out", "in"));

REGISTER_USER_KERNEL("isnan")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kIsNan, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kIsNan, "out", "in"));

}  // namespace user_op
}  // namespace oneflow
