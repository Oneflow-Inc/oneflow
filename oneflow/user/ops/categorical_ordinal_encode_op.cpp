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

REGISTER_USER_OP("CategoricalOrdinalEncode")
    .Input("table")
    .Input("size")
    .Input("in")
    .Output("out")
    .Attr<bool>("hash_precomputed")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const DataType data_type = *ctx->Dtype4ArgNameAndIndex("in", 0);
      CHECK_OR_RETURN(IsIndexDataType(data_type));
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("table", 0), data_type);
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("size", 0), data_type);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = data_type;
      CHECK_EQ_OR_RETURN(ctx->parallel_ctx().parallel_num(), 1);
      const Shape* table_shape = ctx->Shape4ArgNameAndIndex("table", 0);
      CHECK_EQ_OR_RETURN(table_shape->NumAxes(), 1);
      CHECK_EQ_OR_RETURN(table_shape->elem_cnt() % 2, 0);
      const Shape* size_shape = ctx->Shape4ArgNameAndIndex("size", 0);
      CHECK_EQ_OR_RETURN(size_shape->NumAxes(), 1);
      CHECK_EQ_OR_RETURN(size_shape->elem_cnt(), 1);
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* table = GetInputArgModifierFn("table", 0);
      table->set_is_mutable(true);
      table->set_requires_grad(false);
      user_op::InputArgModifier* size = GetInputArgModifierFn("size", 0);
      size->set_is_mutable(true);
      size->set_requires_grad(false);
      user_op::InputArgModifier* in = GetInputArgModifierFn("in", 0);
      in->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->parallel_num(), 1);
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      CHECK_OR_RETURN(op_conf.attr<bool>("hash_precomputed"));
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
