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

/* static */ Maybe<void> CategoricalOrdinalEncodeOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& table_shape = ctx->InputShape("table", 0);
  CHECK_EQ_OR_RETURN(table_shape.NumAxes(), 1);
  CHECK_EQ_OR_RETURN(table_shape.elem_cnt() % 2, 0);
  const Shape& size_shape = ctx->InputShape("size", 0);
  CHECK_EQ_OR_RETURN(size_shape.NumAxes(), 1);
  CHECK_EQ_OR_RETURN(size_shape.elem_cnt(), 1);
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CategoricalOrdinalEncodeOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->parallel_ctx().parallel_num(), 1);
  const Shape& table_shape = ctx->InputShape("table", 0);
  CHECK_EQ_OR_RETURN(table_shape.NumAxes(), 1);
  CHECK_EQ_OR_RETURN(table_shape.elem_cnt() % 2, 0);
  const Shape& size_shape = ctx->InputShape("size", 0);
  CHECK_EQ_OR_RETURN(size_shape.NumAxes(), 1);
  CHECK_EQ_OR_RETURN(size_shape.elem_cnt(), 1);
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CategoricalOrdinalEncodeOp::GetSbp(user_op::SbpContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->parallel_num(), 1);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CategoricalOrdinalEncodeOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* table = GetInputArgModifierFn("table", 0);
  table->set_is_mutable(true);
  table->set_requires_grad(false);
  user_op::InputArgModifier* size = GetInputArgModifierFn("size", 0);
  size->set_is_mutable(true);
  size->set_requires_grad(false);
  user_op::InputArgModifier* in = GetInputArgModifierFn("in", 0);
  in->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CategoricalOrdinalEncodeOp::CheckAttr(
    const user_op::UserOpDefWrapper& def, const user_op::UserOpConfWrapper& conf) {
  CHECK_OR_RETURN(conf.attr<bool>("hash_precomputed"));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CategoricalOrdinalEncodeOp::InferDataType(user_op::InferContext* ctx) {
  DataType data_type = ctx->InputDType("in", 0);
  CHECK_OR_RETURN(IsIndexDataType(data_type));
  CHECK_EQ_OR_RETURN(ctx->InputDType("table", 0), data_type)
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("table", 0))
      << ", but got " << DataType_Name(data_type);
  CHECK_EQ_OR_RETURN(ctx->InputDType("size", 0), data_type)
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("size", 0))
      << ", but got " << DataType_Name(data_type);
  ctx->SetOutputDType("out", 0, data_type);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
