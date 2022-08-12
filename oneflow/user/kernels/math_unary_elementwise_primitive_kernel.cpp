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
#include "oneflow/core/ep/include/primitive/binary_op.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/user/kernels/elementwise_primitive_kernel.h"

namespace oneflow {

REGISTER_USER_KERNEL("exp")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "y", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("y", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("x", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kExp, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kExp, "y", "x"));

// REGISTER_USER_KERNEL("threshold_grad")
//     .SetCreateFn([]() {
//       return user_op::NewOpKernel<BinaryPrimitiveKernel>(
//           "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
//             const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
//             const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
//             return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
//                 ctx->device_type(), ep::primitive::BinaryOp::kThresholdBackwardWithDyX,
//                 src->data_type(), dst->data_type(), 1 /*max_num_dims*/,
//                 ctx->Attr<double>("threshold_val"));
//           });
//     })
//     .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kThresholdBackwardWithDyX,
//     "dx",
//                                            "dy"));

}  // namespace oneflow
