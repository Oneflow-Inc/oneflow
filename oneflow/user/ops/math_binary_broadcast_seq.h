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
#ifndef ONEFLOW_USER_OPS_MATH_BINARY_BROADCAST_SEQ_H_
#define ONEFLOW_USER_OPS_MATH_BINARY_BROADCAST_SEQ_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define MATH_BINARY_BROADCAST_FUNC_SEQ                      \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_add", Add)                \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_sub", Sub)                \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_mul", Mul)                \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_div", Div)                \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_minimum", Min)            \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_maximum", Max)            \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_bitwise_and", BitwiseAnd) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_bitwise_or", BitwiseOr)   \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_bitwise_xor", BitwiseXor) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_floor_mod", FloorMod)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_fmod", FMod)              \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_pow", Pow)

#define MATH_BINARY_BROADCAST_LOGICAL_FUNC_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_equal", EQ)           \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_not_equal", NE)       \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater", GT)         \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater_equal", GE)   \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less", LT)            \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less_equal", LE)      \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_and", AND)    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_or", OR)      \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_xor", XOR)    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_isclose_eq_nan", IEN) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_isclose_neq_nan", INN)

#define MATH_BINARY_BROADCAST_FUNC_SEQ_ODS                \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastAddOp, Add)               \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastSubOp, Sub)               \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastMulOp, Mul)               \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastDivOp, Div)               \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastMinimumOp, Min)           \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastMaximumOp, Max)           \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastBitwiseAndOp, BitwiseAnd) \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastBitwiseOrOp, BitwiseOr)   \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastBitwiseXorOp, BitwiseXor) \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastFloorModOp, FloorMod)     \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastFmodOp, FMod)             \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastPowOp, Pow)

#define MATH_BINARY_BROADCAST_LOGICAL_FUNC_SEQ_ODS      \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastEqualOp, EQ)            \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastNotEqualOp, NE)         \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastGreaterOp, GT)          \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastGreaterEqualOp, GE)     \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLessOp, LT)             \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLessEqualOp, LE)        \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLogicalAndOp, AND)      \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLogicalOrOp, OR)        \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLogicalXorOp, XOR)      \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastIsCloseEqualNanOp, IEN) \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastIsCloseNotEqualNanOp, INN)

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_MATH_BINARY_BROADCAST_SEQ_H_
