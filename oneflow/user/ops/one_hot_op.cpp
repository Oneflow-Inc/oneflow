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
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_USER_OP("one_hot")
    .Input("indices")
    .Output("out")
    .Attr<int64_t>("depth")
    .Attr<double>("floating_on_value")
    .Attr<int64_t>("integer_on_value")
    .Attr<double>("floating_off_value")
    .Attr<int64_t>("integer_off_value")
    .Attr<DataType>("dtype")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const int64_t depth = ctx->Attr<int64_t>("depth");
      CHECK_GT_OR_RETURN(depth, 0);
      const user_op::TensorDesc* indices_desc = ctx->TensorDesc4ArgNameAndIndex("indices", 0);
      CHECK_OR_RETURN(IsIndexDataType(indices_desc->data_type()));
      CHECK_GT_OR_RETURN(indices_desc->shape().NumAxes(), 0);
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc->mut_is_dynamic() = indices_desc->is_dynamic();
      auto dtype = ctx->Attr<DataType>("dtype");
      *out_desc->mut_data_type() = dtype;
      DimVector dim_vec = indices_desc->shape().dim_vec();
      dim_vec.push_back(depth);
      *out_desc->mut_shape() = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
      CHECK(indices_modifier != nullptr);
      indices_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& indices_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
      FOR_RANGE(int64_t, i, 0, indices_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("indices", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }

      return Maybe<void>::Ok();
    })
    .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* indices_desc = ctx->TensorDesc4ArgNameAndIndex("indices", 0);
      CHECK_OR_RETURN(IsIndexDataType(indices_desc->data_type()));
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      DataType dtype = ctx->Attr<DataType>("dtype");
      *out_desc->mut_data_type() = dtype;
      return Maybe<void>::Ok();
    });


}  // namespace oneflow
