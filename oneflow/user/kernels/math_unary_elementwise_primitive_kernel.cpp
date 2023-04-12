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
  OF_PP_MAKE_TUPLE_SEQ("exp2", ep::primitive::UnaryOp::kExp2)                         \
  OF_PP_MAKE_TUPLE_SEQ("expm1", ep::primitive::UnaryOp::kExpm1)                       \
  OF_PP_MAKE_TUPLE_SEQ("floor", ep::primitive::UnaryOp::kFloor)                       \
  OF_PP_MAKE_TUPLE_SEQ("lgamma", ep::primitive::UnaryOp::kLgamma)                     \
  OF_PP_MAKE_TUPLE_SEQ("log", ep::primitive::UnaryOp::kLog)                           \
  OF_PP_MAKE_TUPLE_SEQ("log2", ep::primitive::UnaryOp::kLog2)                         \
  OF_PP_MAKE_TUPLE_SEQ("log10", ep::primitive::UnaryOp::kLog10)                       \
  OF_PP_MAKE_TUPLE_SEQ("log1p", ep::primitive::UnaryOp::kLog1p)                       \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid", ep::primitive::UnaryOp::kLogSigmoid)            \
  OF_PP_MAKE_TUPLE_SEQ("negative", ep::primitive::UnaryOp::kNegative)                 \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal", ep::primitive::UnaryOp::kReciprocal)             \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan", ep::primitive::UnaryOp::kReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ("rint", ep::primitive::UnaryOp::kRint)                         \
  OF_PP_MAKE_TUPLE_SEQ("round", ep::primitive::UnaryOp::kRound)                       \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt", ep::primitive::UnaryOp::kRsqrt)                       \
  OF_PP_MAKE_TUPLE_SEQ("sigmoid", ep::primitive::UnaryOp::kSigmoid)                   \
  OF_PP_MAKE_TUPLE_SEQ("sign", ep::primitive::UnaryOp::kSign)                         \
  OF_PP_MAKE_TUPLE_SEQ("sin", ep::primitive::UnaryOp::kSin)                           \
  OF_PP_MAKE_TUPLE_SEQ("sinh", ep::primitive::UnaryOp::kSinh)                         \
  OF_PP_MAKE_TUPLE_SEQ("sqrt", ep::primitive::UnaryOp::kSqrt)                         \
  OF_PP_MAKE_TUPLE_SEQ("square", ep::primitive::UnaryOp::kSquare)                     \
  OF_PP_MAKE_TUPLE_SEQ("tan", ep::primitive::UnaryOp::kTan)                           \
  OF_PP_MAKE_TUPLE_SEQ("not_equal_zero", ep::primitive::UnaryOp::kNotEqualZero)       \
  OF_PP_MAKE_TUPLE_SEQ("bitwise_not", ep::primitive::UnaryOp::kBitwiseNot)

#define MATH_UNARY_ELEMENTWISE_GRAD_WITH_DY_X_PRIMITIVE_SEQ                                     \
  OF_PP_MAKE_TUPLE_SEQ("abs_grad", ep::primitive::BinaryOp::kAbsBackwardWithDyX)                \
  OF_PP_MAKE_TUPLE_SEQ("acos_grad", ep::primitive::BinaryOp::kAcosBackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("acosh_grad", ep::primitive::BinaryOp::kAcoshBackwardWithDyX)            \
  OF_PP_MAKE_TUPLE_SEQ("asin_grad", ep::primitive::BinaryOp::kAsinBackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("asinh_grad", ep::primitive::BinaryOp::kAsinhBackwardWithDyX)            \
  OF_PP_MAKE_TUPLE_SEQ("atan_grad", ep::primitive::BinaryOp::kAtanBackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("atanh_grad", ep::primitive::BinaryOp::kAtanhBackwardWithDyX)            \
  OF_PP_MAKE_TUPLE_SEQ("cos_grad", ep::primitive::BinaryOp::kCosBackwardWithDyX)                \
  OF_PP_MAKE_TUPLE_SEQ("cosh_grad", ep::primitive::BinaryOp::kCoshBackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("erf_grad", ep::primitive::BinaryOp::kErfBackwardWithDyX)                \
  OF_PP_MAKE_TUPLE_SEQ("erfc_grad", ep::primitive::BinaryOp::kErfcBackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("exp_grad", ep::primitive::BinaryOp::kExpBackwardWithDyX)                \
  OF_PP_MAKE_TUPLE_SEQ("exp2_grad", ep::primitive::BinaryOp::kExp2BackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("expm1_grad", ep::primitive::BinaryOp::kExpm1BackwardWithDyX)            \
  OF_PP_MAKE_TUPLE_SEQ("log_grad", ep::primitive::BinaryOp::kLogBackwardWithDyX)                \
  OF_PP_MAKE_TUPLE_SEQ("lgamma_grad", ep::primitive::BinaryOp::kLgammaBackwardWithDyX)          \
  OF_PP_MAKE_TUPLE_SEQ("log2_grad", ep::primitive::BinaryOp::kLog2BackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("log10_grad", ep::primitive::BinaryOp::kLog10BackwardWithDyX)            \
  OF_PP_MAKE_TUPLE_SEQ("log1p_grad", ep::primitive::BinaryOp::kLog1pBackwardWithDyX)            \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid_grad", ep::primitive::BinaryOp::kLogSigmoidBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_grad", ep::primitive::BinaryOp::kReciprocalBackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan_grad",                                                \
                       ep::primitive::BinaryOp::kReciprocalNoNanBackwardWithDyX)                \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt_grad", ep::primitive::BinaryOp::kRsqrtBackwardWithDyX)            \
  OF_PP_MAKE_TUPLE_SEQ("sin_grad", ep::primitive::BinaryOp::kSinBackwardWithDyX)                \
  OF_PP_MAKE_TUPLE_SEQ("sinh_grad", ep::primitive::BinaryOp::kSinhBackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("sqrt_grad", ep::primitive::BinaryOp::kSqrtBackwardWithDyX)              \
  OF_PP_MAKE_TUPLE_SEQ("square_grad", ep::primitive::BinaryOp::kSquareBackwardWithDyX)          \
  OF_PP_MAKE_TUPLE_SEQ("tan_grad", ep::primitive::BinaryOp::kTanBackwardWithDyX)

#define MATH_UNARY_ELEMENTWISE_GRAD_WITH_DY_Y_PRIMITIVE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ("sigmoid_grad", ep::primitive::BinaryOp::kSigmoidBackwardWithDyY)

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

#define REGISTER_MATH_UNARY_GRAD_PRIMITIVE_WITH_DY_Y_KERNEL(name, BinaryOp)                     \
  REGISTER_USER_KERNEL(name)                                                                    \
      .SetCreateFn([]() {                                                                       \
        return user_op::NewOpKernel<                                                            \
            BinaryPrimitiveKernel>("dx", "dy", "y", [](user_op::KernelComputeContext* ctx) {    \
          const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);            \
          const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);            \
          return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>( \
              ctx->device_type(), BinaryOp, src->data_type(), dst->data_type(),                 \
              1 /*max_num_dims*/);                                                              \
        });                                                                                     \
      })                                                                                        \
      .SetIsMatchedHob(BinaryPrimitiveExists(BinaryOp, "dx", "dy"));

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_GRAD_PRIMITIVE_WITH_DY_Y_KERNEL,
                     MATH_UNARY_ELEMENTWISE_GRAD_WITH_DY_Y_PRIMITIVE_SEQ)

}  // namespace oneflow
