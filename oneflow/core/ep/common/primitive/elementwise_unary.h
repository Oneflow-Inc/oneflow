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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_ELEMENTWISE_UNARY_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_ELEMENTWISE_UNARY_H_

#include "oneflow/core/ep/include/primitive/elementwise_unary.h"

namespace oneflow {

namespace ep {
namespace primitive {

#define UNARY_MATH_OP_SEQ              \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRelu) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIdentity)

#define UNARY_FLOATING_MATH_OP_SEQ                \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kElu)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCelu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kGelu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardSwish)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardSigmoid)     \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardShrink)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardTanh)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLeakyRelu)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kMish)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSelu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSilu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSoftShrink)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSoftSign)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSoftPlus)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kTanh)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kThreshold)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAbs)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAcos)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAcosh)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAsin)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAsinh)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAtan)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAtanh)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCeil)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCos)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCosh)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kErf)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kErfc)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kExp)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kExp2)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kExpm1)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kFloor)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLgamma)          \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog2)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog10)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog1p)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLogSigmoid)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kNegative)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kReciprocal)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRint)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRound)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRsqrt)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSigmoid)         \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSign)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSin)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSinh)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSqrt)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSign)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSquare)          \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kTan)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kTrunc)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kNotEqualZero)    \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kNanAssign)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kFastGelu)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kQuickGelu)

#define UNARY_INT_MATH_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAbs)

#define UNARY_LOGICAL_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLogicalNot)

#define UNARY_BITWISE_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kBitwiseNot)

#define UNARY_UTILS_OP_SEQ              \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIsInf) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIsNan) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIsFinite)

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_ELEMENTWISE_UNARY_H_
