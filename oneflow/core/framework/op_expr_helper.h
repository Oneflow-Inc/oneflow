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

#include "oneflow/core/common/util.h"

namespace oneflow {

namespace one {

class UserOpExpr;

}  // namespace one

namespace op_expr_helper {

Maybe<one::UserOpExpr> AddNOp(int32_t n);

Maybe<one::UserOpExpr> AddOp();

Maybe<one::UserOpExpr> ZeroLikeOp();

Maybe<one::UserOpExpr> OnesLikeOp();

}  // namespace op_expr_helper

}  // namespace oneflow
