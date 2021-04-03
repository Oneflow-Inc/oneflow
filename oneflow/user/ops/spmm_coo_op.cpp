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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4SpmmCOO(user_op::InferContext* ctx) {
  auto a_rows = ctx->Attr<int64_t>("a_rows");
  auto a_cols = ctx->Attr<int64_t>("a_cols");

  CHECK_GT_OR_RETURN(a_rows, 0);
  CHECK_GT_OR_RETURN(a_cols, 0);

  user_op::TensorDesc* b = ctx->TensorDesc4ArgNameAndIndex("b", 0);
  CHECK_GE_OR_RETURN(b->shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(a_cols, b->shape().At(0));

  user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  *out = *b;

  out->mut_shape()->Set(0, a_rows);
  out->mut_shape()->Set(1, b->shape().At(1));

  return Maybe<void>::Ok();
}

REGISTER_USER_OP("spmm_coo")
    .Input("a_cooRowInd")
    .Input("a_cooColInd")
    .Input("a_cooValues")
    .Input("b")
    .Output("out")
    .Attr<int64_t>("a_rows")
    .Attr<int64_t>("a_cols")
    .SetTensorDescInferFn(InferTensorDesc4SpmmCOO);
}  // namespace

}  // namespace oneflow
