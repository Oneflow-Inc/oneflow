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
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/global_for.h"
namespace oneflow {

Maybe<void> InferRandpermNdSbp(user_op::InferNdSbpFnContext* ctx);
REGISTER_NO_GRAD_USER_OP("randperm")
    .Output("out")
    .Attr<int32_t>("n")
    .Attr<int64_t>("seed")
    .Attr<std::vector<std::string>>("nd_sbp")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->OutputShape("out", 0);
      int32_t n = ctx->Attr<int32_t>("n");
      CHECK_GE_OR_RETURN(n, 0);
      *out_shape = Shape({n});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      cfg::SbpParallel default_sbp;
      default_sbp.mutable_broadcast_parallel();
      return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
    });

}  // namespace oneflow
