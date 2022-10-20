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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ auto MegatronGptMmapDataLoaderOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  int64_t batch_size = ctx->Attr<int64_t>("batch_size");
  int64_t sample_len = ctx->Attr<int64_t>("seq_length") + ctx->Attr<int64_t>("label_length");
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_shape(Shape({batch_size, sample_len}));
  return Maybe<void>::Ok();
}
/*static*/ auto MegatronGptMmapDataLoaderOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  ctx->MutOutputTensorDesc("out", 0)->set_data_type(ctx->Attr<DataType>("dtype"));
  return Maybe<void>::Ok();
}
/*static*/ auto MegatronGptMmapDataLoaderOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ auto MegatronGptMmapDataLoaderOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx)
    -> Maybe<void> {
  SbpParallel default_sbp;
  default_sbp.mutable_split_parallel()->set_axis(0);
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}
/*static*/ auto MegatronGptMmapDataLoaderOp::ModifyInputArg(
    const user_op::GetInputArgModifier& GetInputArgModifierFn,
    const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
  if (!conf.has_input("iteration", 0)) { return Maybe<void>::Ok(); }
  user_op::InputArgModifier* input_modifier = GetInputArgModifierFn("iteration", 0);
  CHECK_OR_RETURN(input_modifier != nullptr);
  input_modifier->set_is_mutable(true);
  input_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
