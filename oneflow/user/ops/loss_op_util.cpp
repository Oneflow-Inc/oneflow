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
#include "oneflow/user/ops/loss_op_util.h"
#include "oneflow/core/common/just.h"

namespace oneflow {

user_op::GetSbpFn GenLossForwardDefaultGetSbpFn(
    const std::function<void(user_op::UserOpSbpSignatureBuilder& builder,
                             user_op::SbpContext* ctx)>& f) {
  return [=](user_op::SbpContext* ctx) -> Maybe<void> {
    auto builder = ctx->NewBuilder()
                       .Split(user_op::OpArg("input", 0), 0)
                       .Split(user_op::OpArg("target", 0), 0)
                       .Split(user_op::OpArg("out", 0), 0);
    if (ctx->user_op_conf().has_input("weight", 0)) {
      builder.Split(user_op::OpArg("weight", 0), 0);
    }
    f(builder, ctx);
    builder.Build();
    return Maybe<void>::Ok();
  };
}

user_op::GetSbpFn GenLossBackwardDefaultGetSbpFn(
    const std::function<void(user_op::UserOpSbpSignatureBuilder& builder,
                             user_op::SbpContext* ctx)>& f) {
  return [=](user_op::SbpContext* ctx) -> Maybe<void> {
    auto builder = ctx->NewBuilder()
                       .Split(user_op::OpArg("input", 0), 0)
                       .Split(user_op::OpArg("target", 0), 0)
                       .Split(user_op::OpArg("dx", 0), 0)
                       .Split(user_op::OpArg("dy", 0), 0);
    if (ctx->user_op_conf().has_input("weight", 0)) {
      builder.Split(user_op::OpArg("weight", 0), 0);
    }
    f(builder, ctx);
    builder.Build();
    return Maybe<void>::Ok();
  };
}

}  // namespace oneflow
