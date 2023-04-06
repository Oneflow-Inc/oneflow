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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_BINARY_OP_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_BINARY_OP_H_

#include "oneflow/core/ep/include/primitive/primitive.h"

namespace oneflow {

namespace ep {
namespace primitive {

enum class BinaryOp {
  // Math
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMax,
  kMin,
  kPow,
  kFmod,
  kFloorDiv,
  kTruncDiv,
  kFloorMod,
  kScalarBasePowerGrad,
  kScalarExpPowerGrad,
  // Comparision
  kEqual,
  kNotEqual,
  kLessThan,
  kLessEqual,
  kGreaterThan,
  kGreaterEqual,
  kIsClose,
  kIsCloseEqualNan,
  // Logical
  kLogicalAnd,
  kLogicalOr,
  kLogicalXor,
  // Bitwise
  kBitwiseAnd,
  kBitwiseOr,
  kBitwiseXor,
  // Unary Backward
  kIdentityBackwardWithDyX,
  kEluBackwardWithDyX,
  kCeluBackwardWithDyY,
  kGeluBackwardWithDyX,
  kHardswishBackwardWithDyX,
  kHardsigmoidBackwardWithDyX,
  kHardshrinkBackwardWithDyY,
  kHardtanhBackwardWithDyY,
  kLeakyReluBackwardWithDyX,
  kMishBackwardWithDyX,
  kReluBackwardWithDyY,
  kReluBackwardWithDyX,
  kSeluBackwardWithDyX,
  kSiluBackwardWithDyX,
  kSoftsignBackwardWithDyX,
  kSoftplusBackwardWithDyX,
  kSoftshrinkBackwardWithDyY,
  kTanhBackwardWithDyX,
  kThresholdBackwardWithDyX,
  kSigmoidBackwardWithDyY,
  kSigmoidBackwardWithDyX,
  kAbsBackwardWithDyX,
  kAcosBackwardWithDyX,
  kAcoshBackwardWithDyX,
  kAsinBackwardWithDyX,
  kAsinhBackwardWithDyX,
  kAtanBackwardWithDyX,
  kAtanhBackwardWithDyX,
  kCosBackwardWithDyX,
  kCoshBackwardWithDyX,
  kErfBackwardWithDyX,
  kErfcBackwardWithDyX,
  kExpBackwardWithDyX,
  kExp2BackwardWithDyX,
  kExpm1BackwardWithDyX,
  kLgammaBackwardWithDyX,
  kLogBackwardWithDyX,
  kLog2BackwardWithDyX,
  kLog10BackwardWithDyX,
  kLog1pBackwardWithDyX,
  kLogSigmoidBackwardWithDyX,
  kReciprocalBackwardWithDyX,
  kReciprocalNoNanBackwardWithDyX,
  kRsqrtBackwardWithDyX,
  kSinBackwardWithDyX,
  kSinhBackwardWithDyX,
  kSqrtBackwardWithDyX,
  kSquareBackwardWithDyX,
  kTanBackwardWithDyX,
  kFastGeluBackwardWithDyX,
  kQuickGeluBackwardWithDyX,
};

}
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_BINARY_OP_H_
