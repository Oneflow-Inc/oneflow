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

#define MATH_BINARY_BROADCAST_FUNC_SEQ           \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_add", Add)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_sub", Sub)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_mul", Mul)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_div", Div)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_minimum", Min) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_maximum", Max) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_floor_mod", FloorMod)

#define MATH_BINARY_BROADCAST_LOGICAL_FUNC_SEQ        \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_equal", EQ)         \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_not_equal", NE)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater", GT)       \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater_equal", GE) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less", LT)          \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less_equal", LE)    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_and", AND)

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_MATH_BINARY_BROADCAST_SEQ_H_
