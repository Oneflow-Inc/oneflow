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
  kFloorMod,
  kFMod,
  // Comparision
  kEqual,
  kNotEqual,
  kLessThan,
  kLessEqual,
  kGreaterThan,
  kGreaterEqual,
  // Logical
  kLogicalAnd,
  kLogicalOr,
  kLogicalXor,
  // Unary Backward
  kReluBackwardWithDyY,
  kSigmoidBackwardWithDyY,
  kGeluBackwardWithDyX,

};

}
}  // namespace ep

using namespace ep::primitive;

#define MATH_BINARY_BROADCAST_PRIMITIVE_OP_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_add", BinaryOp::kAdd)            \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_sub", BinaryOp::kSub)            \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_mul", BinaryOp::kMul)            \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_div", BinaryOp::kDiv)            \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_minimum", BinaryOp::kMin)        \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_maximum", BinaryOp::kMax)        \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_floor_mod", BinaryOp::kFloorMod) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_fmod", BinaryOp::kFMod)          \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_pow", BinaryOp::kPow)

// #define MATH_BINARY_BROADCAST_PRIMITIVE_OP_SEQ                  \
//   OF_PP_MAKE_TUPLE_SEQ("broadcast_pow", BinaryOp::kPow)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_BINARY_OP_H_
