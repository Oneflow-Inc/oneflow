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

#define MATH_UNARY_ELEMENTWISE_PRIMITIVE_SEQ                                          \
  OF_PP_MAKE_TUPLE_SEQ("abs", ep::primitive::UnaryOp::kAbs)                           

#define MATH_UNARY_ELEMENTWISE_GRAD_WITH_DY_X_PRIMITIVE_SEQ                                    \
  OF_PP_MAKE_TUPLE_SEQ("abs_grad", ep::primitive::BinaryOp::kAbsBackwardWithDyX)               

#define REGISTER_MATH_UNARY_PRIMITIVE_KERNEL(name, UnaryOp)                               \
  REGISTER_USER_KERNEL(name)                                                              \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "y", "x", [](user_op::KernelComputeContext* ctx) {                            \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("y", 0);   \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("x", 0);   \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), UnaryOp, src->data_type(), dst->data_type());       \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(UnaryOp, "y", "x"));

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_PRIMITIVE_KERNEL, MATH_UNARY_ELEMENTWISE_PRIMITIVE_SEQ)

#define REGISTER_MATH_UNARY_GRAD_PRIMITIVE_WITH_DY_X_KERNEL(name, BinaryOp)                     \
  REGISTER_USER_KERNEL(name)                                                                    \
      .SetCreateFn([]() {                                                                       \
        return user_op::NewOpKernel<                                                            \
            BinaryPrimitiveKernel>("dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {    \
          const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);            \
          const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);            \
          return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>( \
              ctx->device_type(), BinaryOp, src->data_type(), dst->data_type(),                 \
              1 /*max_num_dims*/);                                                              \
        });                                                                                     \
      })                                                                                        \
      .SetIsMatchedHob(BinaryPrimitiveExists(BinaryOp, "dx", "dy"));

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_GRAD_PRIMITIVE_WITH_DY_X_KERNEL,
                     MATH_UNARY_ELEMENTWISE_GRAD_WITH_DY_X_PRIMITIVE_SEQ)


}  // namespace oneflow