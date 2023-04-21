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
// #include "oneflow/core/common/tensor_desc.h"
// #include "oneflow/core/framework/framework.h"
// #include "oneflow/core/framework/op_generated.h"

// namespace oneflow {

// /* static */ Maybe<void> NormalTensorFloatOp::GetSbp(user_op::SbpContext* ctx) {
//   ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
//   return Maybe<void>::Ok();
// }

// /* static */ Maybe<void> NormalFloatTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
// {
//   const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("std", 0);
//   user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("out", 0);
//   y_desc->set_shape(x_desc.shape());
//   return Maybe<void>::Ok();
// }

// /*static*/ Maybe<void> NormalFloatTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
//   return InferLogicalTensorDesc(ctx);
// }

// /* static */ Maybe<void> NormalFloatTensorOp::InferDataType(user_op::InferContext* ctx) {
//   ctx->SetOutputDType("out", 0, ctx->InputDType("std", 0));
//   return Maybe<void>::Ok();
// }

// }  // namespace oneflow
