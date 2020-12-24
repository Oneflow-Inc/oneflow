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
#ifndef ONEFLOW_USER_OPS_MATH_UNARY_ELEMENTWISE_SEQ_H_
#define ONEFLOW_USER_OPS_MATH_UNARY_ELEMENTWISE_SEQ_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define MATH_UNARY_ELEMENTWISE_FUNC_SEQ                      \
  OF_PP_MAKE_TUPLE_SEQ("abs", Abs)                           \
  OF_PP_MAKE_TUPLE_SEQ("acos", Acos)                         \
  OF_PP_MAKE_TUPLE_SEQ("acosh", Acosh)                       \
  OF_PP_MAKE_TUPLE_SEQ("asin", Asin)                         \
  OF_PP_MAKE_TUPLE_SEQ("asinh", Asinh)                       \
  OF_PP_MAKE_TUPLE_SEQ("atan", Atan)                         \
  OF_PP_MAKE_TUPLE_SEQ("atanh", Atanh)                       \
  OF_PP_MAKE_TUPLE_SEQ("ceil", Ceil)                         \
  OF_PP_MAKE_TUPLE_SEQ("cos", Cos)                           \
  OF_PP_MAKE_TUPLE_SEQ("cosh", Cosh)                         \
  OF_PP_MAKE_TUPLE_SEQ("erf", Erf)                           \
  OF_PP_MAKE_TUPLE_SEQ("erfc", Erfc)                         \
  OF_PP_MAKE_TUPLE_SEQ("exp", Exp)                           \
  OF_PP_MAKE_TUPLE_SEQ("expm1", Expm1)                       \
  OF_PP_MAKE_TUPLE_SEQ("floor", Floor)                       \
  OF_PP_MAKE_TUPLE_SEQ("lgamma", Lgamma)                     \
  OF_PP_MAKE_TUPLE_SEQ("log", Log)                           \
  OF_PP_MAKE_TUPLE_SEQ("log1p", Log1p)                       \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid", LogSigmoid)            \
  OF_PP_MAKE_TUPLE_SEQ("negative", Negative)                 \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal", Reciprocal)             \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan", ReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ("rint", Rint)                         \
  OF_PP_MAKE_TUPLE_SEQ("round", Round)                       \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt", Rsqrt)                       \
  OF_PP_MAKE_TUPLE_SEQ("sigmoid_v2", Sigmoid)                \
  OF_PP_MAKE_TUPLE_SEQ("sign", Sign)                         \
  OF_PP_MAKE_TUPLE_SEQ("sin", Sin)                           \
  OF_PP_MAKE_TUPLE_SEQ("sinh", Sinh)                         \
  OF_PP_MAKE_TUPLE_SEQ("softplus", Softplus)                 \
  OF_PP_MAKE_TUPLE_SEQ("sqrt", Sqrt)                         \
  OF_PP_MAKE_TUPLE_SEQ("square", Square)                     \
  OF_PP_MAKE_TUPLE_SEQ("tan", Tan)                           \
  OF_PP_MAKE_TUPLE_SEQ("tanh", Tanh)

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_MATH_UNARY_ELEMENTWISE_SEQ_H_
