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

#define MATH_UNARY_ELEMENTWISE_PRIMIVE_SEQ                      \
  OF_PP_MAKE_TUPLE_SEQ("abs", ep::primitive::UnaryOp::kAbs)                           \
  OF_PP_MAKE_TUPLE_SEQ("acos", ep::primitive::UnaryOp::kAcos)                         \
  OF_PP_MAKE_TUPLE_SEQ("acosh", ep::primitive::UnaryOp::kAcosh)                       \
  OF_PP_MAKE_TUPLE_SEQ("asin", ep::primitive::UnaryOp::kAsin)                         \
  OF_PP_MAKE_TUPLE_SEQ("asinh", ep::primitive::UnaryOp::kAsinh)                       \
  OF_PP_MAKE_TUPLE_SEQ("atan", ep::primitive::UnaryOp::kAtan)                         \
  OF_PP_MAKE_TUPLE_SEQ("atanh", ep::primitive::UnaryOp::kAtanh)                       \
  OF_PP_MAKE_TUPLE_SEQ("ceil", ep::primitive::UnaryOp::kCeil)                         \
  OF_PP_MAKE_TUPLE_SEQ("cos", ep::primitive::UnaryOp::kCos)                           \
  OF_PP_MAKE_TUPLE_SEQ("cosh", ep::primitive::UnaryOp::kCosh)                         \
  OF_PP_MAKE_TUPLE_SEQ("erf", ep::primitive::UnaryOp::kErf)                           \
  OF_PP_MAKE_TUPLE_SEQ("erfc", ep::primitive::UnaryOp::kErfc)                         \
  OF_PP_MAKE_TUPLE_SEQ("exp", ep::primitive::UnaryOp::kExp)                           \
  OF_PP_MAKE_TUPLE_SEQ("expm1", ep::primitive::UnaryOp::kExpm1)                       \
  OF_PP_MAKE_TUPLE_SEQ("floor", ep::primitive::UnaryOp::kFloor)                       \
  OF_PP_MAKE_TUPLE_SEQ("lgamma", ep::primitive::UnaryOp::kLgamma)                     \
  OF_PP_MAKE_TUPLE_SEQ("log", ep::primitive::UnaryOp::kLog)                           \
  OF_PP_MAKE_TUPLE_SEQ("log2", ep::primitive::UnaryOp::kLog2)                         \
  OF_PP_MAKE_TUPLE_SEQ("log1p", ep::primitive::UnaryOp::kLog1p)                       \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid", ep::primitive::UnaryOp::kLogSigmoid)            \
  OF_PP_MAKE_TUPLE_SEQ("negative", ep::primitive::UnaryOp::kNegative)                 \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal", ep::primitive::UnaryOp::kReciprocal)             \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan", ep::primitive::UnaryOp::kReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ("rint", ep::primitive::UnaryOp::kRint)                         \
  OF_PP_MAKE_TUPLE_SEQ("round", ep::primitive::UnaryOp::kRound)                       \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt", ep::primitive::UnaryOp::kRsqrt)                       \
  OF_PP_MAKE_TUPLE_SEQ("sigmoid_v2", ep::primitive::UnaryOp::kSigmoid)                \
  OF_PP_MAKE_TUPLE_SEQ("sign", ep::primitive::UnaryOp::kSign)                         \
  OF_PP_MAKE_TUPLE_SEQ("sin", ep::primitive::UnaryOp::kSin)                           \
  OF_PP_MAKE_TUPLE_SEQ("sinh", ep::primitive::UnaryOp::kSinh)                         \
  OF_PP_MAKE_TUPLE_SEQ("sqrt", ep::primitive::UnaryOp::kSqrt)                         \
  OF_PP_MAKE_TUPLE_SEQ("square", ep::primitive::UnaryOp::kSquare)                     \
  OF_PP_MAKE_TUPLE_SEQ("tan", ep::primitive::UnaryOp::kTan)                           \
  OF_PP_MAKE_TUPLE_SEQ("not_equal_zero", ep::primitive::UnaryOp::kNotEqualZero)


// REGISTER_USER_KERNEL("exp")
//     .SetCreateFn([]() {
//       return user_op::NewOpKernel<UnaryPrimitiveKernel>(
//           "y", "x", [](user_op::KernelComputeContext* ctx) {
//             const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("y", 0);
//             const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("x", 0);
//             return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
//                 ctx->device_type(), ep::primitive::UnaryOp::kExp, src->data_type(),
//                 dst->data_type());
//           });
//     })
//     .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kExp, "y", "x"));

#define REGISTER_MATH_UNARY_PRIMITIVE_KERNEL(name, UnaryOp) \
    REGISTER_USER_KERNEL(name) \
    .SetCreateFn([]() { \
      return user_op::NewOpKernel<UnaryPrimitiveKernel>( \
          "y", "x", [](user_op::KernelComputeContext* ctx) { \
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("y", 0); \
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("x", 0); \
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                ctx->device_type(), UnaryOp, src->data_type(), \
                dst->data_type()); \
          }); \
    }) \
    .SetIsMatchedHob(UnaryPrimitiveExists(UnaryOp, "y", "x"));

// OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_PRIMITIVE_KERNEL, MATH_UNARY_ELEMENTWISE_PRIMIVE_SEQ)
REGISTER_MATH_UNARY_PRIMITIVE_KERNEL("abs", ep::primitive::UnaryOp::kAbs)




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
//     .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kThresholdBackwardWithDyX, "dx",
//                                            "dy"));

}  // namespace oneflow
