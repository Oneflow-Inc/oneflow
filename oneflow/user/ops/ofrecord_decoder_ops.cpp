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

REGISTER_CPU_ONLY_USER_OP("ofrecord_raw_decoder")
    .Input("in")
    .Output("out")
    .Attr<std::string>("name")
    .Attr<Shape>("shape")
    .Attr<DataType>("data_type")
    .Attr<bool>("dim1_varying_length", false)
    .Attr<bool>("auto_zero_padding", false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kOFRecord);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().At(0) >= 1);
      Shape conf_shape = ctx->Attr<Shape>("shape");
      DimVector dim_vec(1 + conf_shape.NumAxes());
      dim_vec[0] = in_tensor->shape().At(0);
      for (int i = 1; i < dim_vec.size(); ++i) { dim_vec[i] = conf_shape.At(i - 1); }
      *out_tensor->mut_shape() = Shape(dim_vec);
      *out_tensor->mut_data_type() = ctx->Attr<DataType>("data_type");
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
      CHECK_NOTNULL(in_modifier);
      in_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("ofrecord_bytes_decoder")
    .Input("in")
    .Output("out")
    .Attr<std::string>("name")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in->data_type() == DataType::kOFRecord);
      *out = *in;
      *out->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
      CHECK_NOTNULL(in_modifier);
      in_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis);

REGISTER_CPU_ONLY_USER_OP("ofrecord_image_decoder")
    .Input("in")
    .Output("out")
    .Attr<std::string>("name")
    .Attr<std::string>("color_space", "BGR")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kOFRecord);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().At(0) >= 1);
      *out_tensor->mut_shape() = in_tensor->shape();
      *out_tensor->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
      CHECK_NOTNULL(in_modifier);
      in_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("ofrecord_image_decoder_random_crop")
    .Input("in")
    .Output("out")
    .Attr<std::string>("name")
    .Attr<std::string>("color_space", "BGR")
    .Attr<int32_t>("num_attempts", 10)
    .Attr<int64_t>("seed", -1)
    .Attr<bool>("has_seed", false)
    .Attr<std::vector<float>>("random_area", {0.08, 1.0})
    .Attr<std::vector<float>>("random_aspect_ratio", {0.75, 1.333333})
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kOFRecord);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().At(0) >= 1);
      *out_tensor->mut_shape() = in_tensor->shape();
      *out_tensor->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
      CHECK_NOTNULL(in_modifier);
      in_modifier->set_requires_grad(false);
    });

}  // namespace oneflow
