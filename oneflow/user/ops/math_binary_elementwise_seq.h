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
#ifndef ONEFLOW_USER_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_
#define ONEFLOW_USER_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define MATH_BINARY_ELEMENTWISE_FUNC_SEQ     \
  OF_PP_MAKE_TUPLE_SEQ("pow", Pow)           \
  OF_PP_MAKE_TUPLE_SEQ("atan2", Atan2)       \
  OF_PP_MAKE_TUPLE_SEQ("floordiv", FloorDiv) \
  OF_PP_MAKE_TUPLE_SEQ("truncdiv", TruncDiv) \
  OF_PP_MAKE_TUPLE_SEQ("xdivy", Xdivy)       \
  OF_PP_MAKE_TUPLE_SEQ("xlogy", Xlogy)

#define MATH_BINARY_ELEMENTWISE_FUNC_SEQ_ODS \
  OF_PP_MAKE_TUPLE_SEQ("pow", Pow)           \
  OF_PP_MAKE_TUPLE_SEQ("atan2", Atan2)       \
  OF_PP_MAKE_TUPLE_SEQ("floordiv", Floordiv) \
  OF_PP_MAKE_TUPLE_SEQ("truncdiv", Truncdiv) \
  OF_PP_MAKE_TUPLE_SEQ("xdivy", Xdivy)       \
  OF_PP_MAKE_TUPLE_SEQ("xlogy", Xlogy)

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_
