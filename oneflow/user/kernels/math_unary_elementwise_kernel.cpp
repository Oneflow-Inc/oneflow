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
#include "oneflow/user/ops/math_unary_elementwise_seq.h"
#include "oneflow/user/kernels/elementwise_primitive_kernel.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::ElementwiseUnary> NewElementwiseUnaryPrimitive(Context* ctx, ep::primitive::UnaryOp unary_op) {
  const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnary>(
      ctx->device_type(), unary_op, src->data_type(), dst->data_type(), 1);
}

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewElementwiseBinaryPrimitive(Context* ctx, ep::primitive::BinaryOp binary_op) {
  const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinary>(
      ctx->device_type(), binary_op, src->data_type(), dst->data_type(), 1);
}

}  // namespace

#define REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD(math_type_pair)        \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                           \
      .SetCreateFn([]() {\
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(\
            "y", "x", [](user_op::KernelComputeContext* ctx) {\
              return NewElementwiseUnaryPrimitive<user_op::KernelComputeContext>(ctx, OF_PP_CAT(ep::primitive::UnaryOp::k, OF_PP_PAIR_SECOND(math_type_pair)));\
            });\
      })                       \
      .SetIsMatchedHob(UnaryPrimitiveExists(OF_PP_CAT(ep::primitive::UnaryOp::k, OF_PP_PAIR_SECOND(math_type_pair)));    \
                                                                                                   \
  REGISTER_USER_KERNEL((std::string("") + OF_PP_PAIR_FIRST(math_type_pair) + "_grad"))             \
      .SetCreateFn([]() {\
        return user_op::NewOpKernel<BinaryPrimitiveKernel>(\
            "dx", "dy", [](user_op::KernelComputeContext* ctx) {\
              return NewElementwiseBinaryPrimitive<user_op::KernelComputeContext>(ctx, OF_PP_CAT(OF_PP_CAT(ep::primitive::BinaryOp::k, OF_PP_PAIR_SECOND(math_type_pair)), BackwardWithDyX)); \
            });\
      })                   \
      .SetIsMatchedHob(UnaryPrimitiveExists(OF_PP_CAT(OF_PP_CAT(ep::primitive::BinaryOp::k, OF_PP_PAIR_SECOND(math_type_pair)), BackwardWithDyX));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD,
                                 MATH_UNARY_ELEMENTWISE_FUNC_SEQ)

}  // namespace oneflow
