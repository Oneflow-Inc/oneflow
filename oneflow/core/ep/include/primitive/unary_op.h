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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_UNARY_OP_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_UNARY_OP_H_

namespace oneflow {

namespace ep {
namespace primitive {

enum class UnaryOp {
  kIdentity,
  // activation op
  kElu,
  kCelu,
  kRelu,
  kGelu,
  kHardSwish,
  kHardSigmoid,
  kHardShrink,
  kHardTanh,
  kLeakyRelu,
  kMish,
  kSelu,
  kSilu,
  kSoftShrink,
  kSoftSign,
  kSoftPlus,
  kTanh,
  kThreshold,
  kFastGelu,
  kQuickGelu,
  // math op
  kAbs,
  kAcos,
  kAcosh,
  kAsin,
  kAsinh,
  kAtan,
  kAtanh,
  kCeil,
  kCos,
  kCosh,
  kErf,
  kErfc,
  kExp,
  kExp2,
  kExpm1,
  kFloor,
  kLgamma,
  kLog,
  kLog2,
  kLog10,
  kLog1p,
  kLogSigmoid,
  kNegative,
  kReciprocal,
  kReciprocalNoNan,
  kRint,
  kRound,
  kRsqrt,
  kSigmoid,
  kSign,
  kSin,
  kSinh,
  kSqrt,
  kSquare,
  kTan,
  kTrunc,
  kNotEqualZero,
  // logical op
  kLogicalNot,

  // cast op
  kCast,

  // utils op
  kIsInf,
  kIsNan,
  kIsFinite,
  kNanAssign,

  // bitwise op
  kBitwiseNot,
};

}
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_UNARY_OP_H_
