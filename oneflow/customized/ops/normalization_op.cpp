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

Maybe<void> NormalizationTensorDescInfer(user_op::InferContext* ctx) {
#ifdef WITH_CUDA
  // assume cudnn is enabled
  CHECK_GE_OR_RETURN(ctx->Attr<float>("epsilon"), CUDNN_BN_MIN_EPSILON);
#endif
  const auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const auto data_type = x->data_type();
  *ctx->TensorDesc4ArgNameAndIndex("y", 0) = *x;
  const auto axis = ctx->Attr<int32_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LT_OR_RETURN(axis, x->shape().NumAxes());
  const Shape param_shape({x->shape().At(axis)});
  const DataType param_data_type = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
  const auto CheckParamTensorDesc = [&](const std::string& bn) -> Maybe<void> {
    const auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
    if (tensor_desc != nullptr) {
      CHECK_EQ_OR_RETURN(tensor_desc->data_type(), param_data_type);
      CHECK_EQ_OR_RETURN(tensor_desc->shape(), param_shape);
    }
    return Maybe<void>::Ok();
  };
  const auto SetParamTensorDesc = [&](const std::string& bn) -> Maybe<void> {
    auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
    CHECK_OR_RETURN(tensor_desc != nullptr);
    *tensor_desc->mut_data_type() = param_data_type;
    *tensor_desc->mut_shape() = param_shape;
    return Maybe<void>::Ok();
  };
  JUST(CheckParamTensorDesc("moving_mean"));
  JUST(CheckParamTensorDesc("moving_variance"));
  JUST(CheckParamTensorDesc("beta"));
  JUST(CheckParamTensorDesc("gamma"));
  if (ctx->Attr<bool>("training")) {
    JUST(SetParamTensorDesc("mean"));
    JUST(SetParamTensorDesc("inv_variance"));
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("normalization")
    .Input("x")
    .Input("moving_mean")
    .Input("moving_variance")
    .Input("gamma")
    .Input("beta")
    .Output("y")
    .OptionalOutput("mean")
    .OptionalOutput("inv_variance")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .Attr("training", UserOpAttrType::kAtBool)
    .Attr("momentum", UserOpAttrType::kAtFloat)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) {
      user_op::InputArgModifier* moving_mean_modifier = GetInputArgModifierFn("moving_mean", 0);
      CHECK(moving_mean_modifier != nullptr);
      const bool training = conf.attr<bool>("training");
      moving_mean_modifier->set_is_mutable(training);
      moving_mean_modifier->set_requires_grad(false);
      user_op::InputArgModifier* moving_variance_modifier =
          GetInputArgModifierFn("moving_variance", 0);
      CHECK(moving_variance_modifier != nullptr);
      moving_variance_modifier->set_is_mutable(training);
      moving_variance_modifier->set_requires_grad(false);
    })
    .SetTensorDescInferFn(NormalizationTensorDescInfer)
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      if (ctx->Attr<bool>("training")) {
        ctx->BatchAxis4ArgNameAndIndex("mean", 0)->clear_value();
        ctx->BatchAxis4ArgNameAndIndex("inv_variance", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Broadcast(ctx->inputs())
          .Broadcast(ctx->outputs())
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

Maybe<void> NormalizationGradTensorDescInfer(user_op::InferContext* ctx) {
#ifdef WITH_CUDA
  // assume cudnn is enabled
  CHECK_GE_OR_RETURN(ctx->Attr<float>("epsilon"), CUDNN_BN_MIN_EPSILON);
#endif
  const auto x_type = *ctx->Dtype4ArgNameAndIndex("x", 0);
  const auto dy_type = *ctx->Dtype4ArgNameAndIndex("dy", 0);
  CHECK_EQ_OR_RETURN(x_type, dy_type);
  const auto x_shape = *ctx->Shape4ArgNameAndIndex("x", 0);
  const auto dy_shape = *ctx->Shape4ArgNameAndIndex("dy", 0);
  CHECK_EQ_OR_RETURN(dy_shape, x_shape);
  *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *ctx->TensorDesc4ArgNameAndIndex("x", 0);

  const Shape param_shape({x_shape.At(ctx->Attr<int32_t>("axis"))});
  const DataType param_data_type = x_type == DataType::kFloat16 ? DataType::kFloat : x_type;
  const auto CheckParamBlobDesc = [&](const std::string& bn) -> Maybe<void> {
    CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex(bn, 0), param_data_type);
    const auto* blob_shape = ctx->Shape4ArgNameAndIndex(bn, 0);
    if (blob_shape) { CHECK_EQ_OR_RETURN(*blob_shape, param_shape); }
    return Maybe<void>::Ok();
  };
  const auto SetParamBlobDesc = [&](const std::string& bn) -> Maybe<void> {
    auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
    if (tensor_desc) {
      *tensor_desc->mut_data_type() = param_data_type;
      *tensor_desc->mut_shape() = param_shape;
    }
    return Maybe<void>::Ok();
  };

  JUST(CheckParamBlobDesc("mean"));
  JUST(CheckParamBlobDesc("inv_variance"));
  JUST(CheckParamBlobDesc("gamma"));
  JUST(SetParamBlobDesc("gamma_diff"));
  JUST(SetParamBlobDesc("beta_diff"));
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("normalization_grad")
    .Input("x")
    .Input("dy")
    .Input("mean")
    .Input("inv_variance")
    .Input("gamma")
    .Output("gamma_diff")
    .Output("beta_diff")
    .Output("dx")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn(NormalizationGradTensorDescInfer)
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("dy", 0);
      ctx->BatchAxis4ArgNameAndIndex("gamma_diff", 0)->clear_value();
      ctx->BatchAxis4ArgNameAndIndex("beta_diff", 0)->clear_value();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Broadcast(ctx->inputs())
          .PartialSum(ctx->outputs())
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("dx", 0), 0)
          .Split(user_op::OpArg("dy", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("normalization")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const bool is_training = ctx->FwOp().attr<bool>("training");
      const bool is_fp16 =
          ctx->FwOp().TensorDesc4ArgNameAndIndex("y", 0).data_type() == DataType::kFloat16;

      const auto var_add_eps_op_name =
          "System-AutoGrad-" + ctx->FwOp().op_name() + "-VarianceAddEpsilon";
      ctx->DefineOp(var_add_eps_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
        return builder.OpType("scalar_add")
            .InputBind("in", ctx->FwOp().input("moving_variance", 0))
            .Attr("has_float_operand", true)
            .Attr("has_int_operand", false)
            .Attr("int_operand", static_cast<int64_t>(0))
            .Attr("float_operand", static_cast<double>(ctx->FwOp().attr<float>("epsilon")))
            .Output("out")
            .Finish();
      });

      const atuo var_rsqrt_op_name = "System-AutoGrad-" + ctx->FwOp().op_name() + "-VarianceRsqrt";
      ctx->DefineOp(var_rsqrt_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
        return builder.OpTypeName("rsqrt")
            .InputBind("x", ctx->GetOp(var_add_eps_op_name).output("out", 0))
            .Output("y")
            .Finish();
      });

      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(grad_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
        builder.OpTypeName("normalization_grad")
            .InputBind("x", ctx->FwOp().input("x", 0))
            .InputBind("dy", ctx->FwOp().input_grad("y", 0))
            .InputBind("gamma", ctx->FwOp().input("gamma", 0))
            .Attr("axis", ctx->FwOp().attr<int32_t>("axis"))
            .Attr("epsilon", ctx->FwOp().attr<float>("epsilon"))
            .Output("gamma_diff")
            .Output("beta_diff")
            .Output("dx");
        if (is_training) {
          builder.InputBind("mean", ctx->FwOp().output("mean", 0))
              .InputBind("inv_variance", ctx->FwOp().output("inv_variance", 0));
        } else {
          builder.InputBind("mean", ctx->FwOp().input("moving_mean", 0))
              .InputBind("inv_variance", ctx->GetOp(var_rsqrt_op_name).output("y", 0));
        }
        return builder.Finish();
      });

      // calculate dx manually as cudnn cannot be used in evaluation mode
      // reference: https://github.com/pytorch/pytorch/issues/4284
      const auto axis = ctx->FwOp().attr<int32_t>("axis");
      const auto BroadcastMulAtAxisOpDefine =
          [&axis, &ctx](std::function<std::string()> scale_bn_func,
                        std::function<std::string()> input_bn_func, const std::string& name) {
            DimVector broadcast_dim_vec;
            const auto& in_shape = ctx->FwOp().TensorDesc4ArgNameAndIndex("x", 0).shape();
            FOR_RANGE(size_t, i, 0, in_shape.NumAxes()) {
              if (i != axis) {
                broadcast_dim_vec.push_back(1);
              } else {
                broadcast_dim_vec.push_back(in_shape.At(axis));
              }
            }
            const Shape broadcast_shape(broadcast_dim_vec);

            const auto reshape_op_name = "System-AutoGrad-" + name + "-Reshape";
            ctx->DefineOp(reshape_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
              return builder.OpTypeName("reshape")
                  .InputBind("in", scale_bn_func())
                  .Attr("shape", broadcast_shape)
                  .Output("out")
                  .Finish();
            });

            const auto mul_op_name = "System-AutoGrad-" + name + "-BroadcastMul";
            ctx->DefineOp(mul_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
              return builder.OpTypeName("broadcast_mul")
                  .InputBind("x", ctx->GetOp(reshape_op_name).output("out", 0))
                  .InputBind("y", input_bn_func())
                  .Output("z")
                  .Finish();
            });
          };

      const auto dy_h2f_cast_op_name = "System-AutoGrad-" + ctx->FwOp().op_name() + "-Cast-dy-h2f";
      ctx->DefineOp(dy_h2f_cast_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
        return builder.OpTypeName("cast")
            .Input("in", ctx->FwOp().output_grad("y", 0))
            .Output("out")
            .Attr("dtype", ctx->FwOp().TensorDesc4ArgNameAndIndex("gamma", 0).data_type())
            .Finish();
      });

      const auto mul_gamma_name = "out_grad_mul_gamma";
      const auto dy_mul_gamma_op_name = "System-AutoGrad-" + mul_gamma_name + "-BroadcastMul";
      BroadcastMulAtAxisOpDefine([&]() { return ctx->FwOp().input("gamma", 0); },
                                 [&]() {
                                   if (is_fp16) {
                                     return ctx->GetOp(dy_h2f_cast_op_name).output("out", 0);
                                   } else {
                                     return ctx->FwOp().output_grad("y", 0);
                                   }
                                 },
                                 mul_gamma_name);

      const auto mul_inv_var_name = "out_grad_mul_inv_var";
      const auto dy_mul_inv_var_op_name = "System-AutoGrad-" + mul_inv_var_name + "-BroadcastMul";
      BroadcastMulAtAxisOpDefine([&]() { return ctx->GetOp(var_rsqrt_op_name).output("y", 0); },
                                 [&]() { return ctx->GetOp(dy_mul_gamma_op_name).output("z", 0); },
                                 mul_inv_var_name);

      const auto dx_f2h_cast_op_name = "System-AutoGrad-" + ctx->FwOp().op_name() + "-Cast-dx-f2h";
      ctx->DefineOp(dx_f2h_cast_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
        return builder.OpTypeName("cast")
            .InputBind("in", ctx->GetOp(dy_mul_inv_var_op_name).output("z", 0))
            .Output("out")
            .Attr("dtype", DataType::kFloat16)
            .Finish();
      });

      // TODO(liujuncheng): delete identity op when boxing support separated regsts
      const auto gamma_identity_op_name = ctx->FwOp().op_name() + "_grad_gamma_diff_identity";
      ctx->DefineOp(gamma_identity_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
        return builder.OpTypeName("identity")
            .InputBind("in", ctx->GetOp(grad_op_name).output("gamma_diff", 0))
            .Output("out")
            .Finish();
      });

      // TODO(liujuncheng): delete identity op when boxing support separated regsts
      const auto beta_identity_op_name = ctx->FwOp().op_name() + "_grad_beta_diff_identity";
      ctx->DefineOp(beta_identity_op_name, [&](user_op::UserOpConfWrapperBuilder& builder) {
        return builder.OpTypeName("identity")
            .InputBind("in", ctx->GetOp(grad_op_name).output("beta_diff", 0))
            .Output("out")
            .Finish();
      });

      ctx->FwOp().InputGradBind(OpArg("x", 0), [&]() {
        if (is_training) {
          return ctx->GetOp(grad_op_name).output("dx", 0);
        } else {
          if (is_fp16) {
            return ctx->GetOp(dx_f2h_cast_op_name).output("out", 0);
          } else {
            return ctx->GetOp(dy_mul_inv_var_op_name).output("z", 0);
          }
        }
      });

      ctx->FwOp().InputGradBind(
          OpArg("gamma", 0), [&]() { return ctx->GetOp(gamma_identity_op_name).output("out", 0); });
      ctx->FwOp().InputGradBind(
          OpArg("beta", 0), [&]() { return ctx->GetOp(beta_identity_op_name).output("out", 0); });
    });

}  // namespace oneflow
