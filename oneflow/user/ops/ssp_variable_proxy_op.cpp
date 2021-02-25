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

REGISTER_USER_OP("ssp_variable_proxy")
    .Input("var")
    .Output("ref")
    .Output("value")
    .Attr<int64_t>("buffer_size", 1)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* var_shape = ctx->Shape4ArgNameAndIndex("var", 0);
      *ctx->Shape4ArgNameAndIndex("ref", 0) = *var_shape;
      *ctx->Shape4ArgNameAndIndex("value", 0) = *var_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& var_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("var", 0);
      FOR_RANGE(int64_t, i, 0, var_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("var", 0), i)
            .Split(user_op::OpArg("ref", 0), i)
            .Split(user_op::OpArg("value", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) {
      user_op::OutputArgModifier* out_modifier = GetOutputArgModifierFn("ref", 0);
      CHECK(out_modifier != nullptr);
      out_modifier->set_is_mutable(true);
    });

}  // namespace

}  // namespace oneflow
