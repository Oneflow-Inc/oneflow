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

REGISTER_USER_OP("fake_quantization")
    .Input("in")
    .Input("scale")
    .Input("zero_point")
    .Output("out")
    // NOTE(Liang Depeng): quantize from float32 to "quantize_to_bit" bit signed or unsigned integer
    .Attr<int32_t>("quantize_to_bit", 8)
    // NOTE(Liang Depeng): "symmetric" or "affine": quantize to signed or unsigned integer
    .Attr<std::string>("quantizer_type", "symmetric")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* scale_shape = ctx->Shape4ArgNameAndIndex("scale", 0);
      Shape* zero_point_shape = ctx->Shape4ArgNameAndIndex("zero_point", 0);

      std::string quantizer_type = ctx->Attr<std::string>("quantizer_type");

      // NOTE(Liang Depeng): scale_shape->elem_cnt() > 1 means per-channel quantization for weights.
      if (scale_shape->elem_cnt() > 1) {
        CHECK_OR_RETURN(scale_shape->elem_cnt() == in_shape->At(0));
        CHECK_OR_RETURN(zero_point_shape->elem_cnt() == in_shape->At(0));
      }

      *ctx->Shape4ArgNameAndIndex("out", 0) = *in_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* scale = GetInputArgModifierFn("scale", 0);
      CHECK(scale != nullptr);
      scale->set_requires_grad(false);

      user_op::InputArgModifier* zero_point = GetInputArgModifierFn("zero_point", 0);
      CHECK(zero_point != nullptr);
      zero_point->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO(Liang Depeng): refer to eltwise op
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      int32_t quantize_to_bit = op_conf.attr<int32_t>("quantize_to_bit");
      CHECK_GT_OR_RETURN(quantize_to_bit, 1);
      CHECK_LE_OR_RETURN(quantize_to_bit, 8);

      std::string quantizer_type = op_conf.attr<std::string>("quantizer_type");
      CHECK_OR_RETURN(quantizer_type == "symmetric" || quantizer_type == "affine");
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("fake_quantization")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper identity_op =
            builder.Op("identity")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(identity_op.output("out", 0), "in", 0);
        AddOp(identity_op);
      }
    });

}  // namespace

}  // namespace oneflow
