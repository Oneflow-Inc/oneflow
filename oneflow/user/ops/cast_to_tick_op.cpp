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
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("cast_to_tick")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = Shape({1});
      return Maybe<void>::Ok();
    })
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      SbpSignature* signature = ctx->mutable_sbp_signature();
      auto* bn2sbp = signature->mutable_bn_in_op2sbp_parallel();
      (*bn2sbp)[GenRepeatedBn("in", 0)] = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
      (*bn2sbp)[GenRepeatedBn("out", 0)].mutable_broadcast_parallel();
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
