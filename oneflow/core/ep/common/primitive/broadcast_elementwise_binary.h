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
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_BINARY
#define ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_BINARY

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/binary_op.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ep/common/primitive/util.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace broadcast_elementwise_binary {

constexpr size_t kMaxNumDims = 8;

inline bool IsDimsEquals(size_t num_src0_dims, const int64_t* src0_dims, size_t num_src1_dims,
                         const int64_t* src1_dims) {
  if (num_src0_dims != num_src1_dims) { return false; }
  for (size_t i = 0; i < num_src1_dims; ++i) {
    if (src0_dims[i] != src1_dims[i]) { return false; }
  }
  return true;
}

#define BINARY_MATH_OP_SEQ_0           \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSub) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMul) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kDiv) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMax)

#define BINARY_MATH_OP_SEQ_1                \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMin)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kPow)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kFmod)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kFloorDiv) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kTruncDiv)

#define BINARY_MATH_OP_SEQ_2                           \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kFloorMod)            \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kScalarBasePowerGrad) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kScalarExpPowerGrad)

#define BINARY_MATH_OP_SEQ \
  BINARY_MATH_OP_SEQ_0     \
  BINARY_MATH_OP_SEQ_1     \
  BINARY_MATH_OP_SEQ_2

#define BINARY_COMPARISION_OP_SEQ_0         \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEqual)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kNotEqual) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessThan) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessEqual)

#define BINARY_COMPARISION_OP_SEQ_1                \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterThan)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterEqual)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kIsCloseEqualNan) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kIsClose)

#define BINARY_COMPARISION_OP_SEQ \
  BINARY_COMPARISION_OP_SEQ_0     \
  BINARY_COMPARISION_OP_SEQ_1

#define BINARY_LOGICAL_OP_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalAnd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalOr)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalXor)

#define BINARY_BITWISE_OP_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kBitwiseAnd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kBitwiseOr)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kBitwiseXor)

#define BINARY_ACTIVATION_BACKWARD_OP_SEQ_0                   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kIdentityBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEluBackwardWithDyX)         \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kCeluBackwardWithDyY)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGeluBackwardWithDyX)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardswishBackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardsigmoidBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardshrinkBackwardWithDyY)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardtanhBackwardWithDyY)

#define BINARY_ACTIVATION_BACKWARD_OP_SEQ_1                 \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLeakyReluBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMishBackwardWithDyX)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kReluBackwardWithDyY)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kReluBackwardWithDyX)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSeluBackwardWithDyX)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSiluBackwardWithDyX)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSoftsignBackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSoftplusBackwardWithDyX)

#define BINARY_ACTIVATION_BACKWARD_OP_SEQ_2                  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSoftshrinkBackwardWithDyY) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kTanhBackwardWithDyX)       \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kThresholdBackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kFastGeluBackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kQuickGeluBackwardWithDyX)

#define BINARY_ACTIVATION_BACKWARD_OP_SEQ \
  BINARY_ACTIVATION_BACKWARD_OP_SEQ_0     \
  BINARY_ACTIVATION_BACKWARD_OP_SEQ_1     \
  BINARY_ACTIVATION_BACKWARD_OP_SEQ_2

#define BINARY_MATH_BACKWARD_OP_SEQ_0                   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAbsBackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAcosBackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAcoshBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAsinBackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAsinhBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAtanBackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAtanhBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kCosBackwardWithDyX)

#define BINARY_MATH_BACKWARD_OP_SEQ_1                    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kCoshBackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kErfBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kErfcBackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kExpBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kExp2BackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kExpm1BackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLgammaBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLog2BackwardWithDyX)

#define BINARY_MATH_BACKWARD_OP_SEQ_2                             \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLog10BackwardWithDyX)           \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLog1pBackwardWithDyX)           \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogSigmoidBackwardWithDyX)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kReciprocalBackwardWithDyX)      \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kReciprocalNoNanBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kRsqrtBackwardWithDyX)           \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSinBackwardWithDyX)             \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSigmoidBackwardWithDyY)

#define BINARY_MATH_BACKWARD_OP_SEQ_3                     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSigmoidBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSinhBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSqrtBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSquareBackwardWithDyX)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kTanBackwardWithDyX)

#define BINARY_MATH_BACKWARD_OP_SEQ \
  BINARY_MATH_BACKWARD_OP_SEQ_0     \
  BINARY_MATH_BACKWARD_OP_SEQ_1     \
  BINARY_MATH_BACKWARD_OP_SEQ_2     \
  BINARY_MATH_BACKWARD_OP_SEQ_3

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_BINARY
