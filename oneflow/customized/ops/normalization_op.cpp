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
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0) || op.NeedGenGradTensor4OpInput("gamma", 0)
          || op.NeedGenGradTensor4OpInput("beta", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        auto grad_op_builder = builder.Op("normalization_grad")
                                   .Input("x", op.input("x", 0))
                                   .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                   .Input("gamma", op.input("gamma", 0))
                                   .Attr("axis", op.attr<int32_t>("axis"))
                                   .Attr("epsilon", op.attr<float>("epsilon"))
                                   .Output("gamma_diff")
                                   .Output("beta_diff")
                                   .Output("dx");
        const bool training = op.attr<bool>("training");
        if (training) {
          grad_op_builder.Input("mean", op.output("mean", 0))
              .Input("inv_variance", op.output("inv_variance", 0));
        } else {
          grad_op_builder.Input("mean", op.input("moving_mean", 0));

          // calculate inv_variance from moving_variance
          const auto var_add_eps_op_name =
              "System-AutoGrad-" + op.op_name() + "-VarianceAddEpsilon";
          const auto var_add_eps_op =
              user_op::UserOpConfWrapperBuilder(var_add_eps_op_name)
                  .Op("scalar_add")
                  .Input("in", op.input("moving_variance", 0))
                  .Attr("has_float_operand", true)
                  .Attr("has_int_operand", false)
                  .Attr("int_operand", static_cast<int64_t>(0))
                  .Attr("float_operand", static_cast<double>(op.attr<float>("epsilon")))
                  .Output("out")
                  .Build();
          AddOp(var_add_eps_op);

          const auto var_rsqrt_op_name = "System-AutoGrad-" + op.op_name() + "-VarianceRsqrt";
          const auto var_sqrt_op = user_op::UserOpConfWrapperBuilder(var_rsqrt_op_name)
                                       .Op("rsqrt")
                                       .Input("x", var_add_eps_op.output("out", 0))
                                       .Output("y")
                                       .Build();
          AddOp(var_sqrt_op);

          grad_op_builder.Input("inv_variance", var_sqrt_op.output("y", 0));

          if (op.NeedGenGradTensor4OpInput("x", 0)) {
            // calculate dx manually as cudnn cannot be used in evaluation mode
            // reference: https://github.com/pytorch/pytorch/issues/4284
            const auto axis = op.attr<int32_t>("axis");
            const auto BroadcastMulAtAxis = [&op, &axis, &AddOp](const std::string& scale_bn,
                                                                 const std::string& input_bn,
                                                                 const std::string& name) {
              DimVector broadcast_dim_vec;
              const auto& in_shape = op.TensorDesc4ArgNameAndIndex("x", 0).shape();
              FOR_RANGE(size_t, i, 0, in_shape.NumAxes()) {
                if (i != axis) {
                  broadcast_dim_vec.push_back(1);
                } else {
                  broadcast_dim_vec.push_back(in_shape.At(axis));
                }
              }
              const Shape broadcast_shape(broadcast_dim_vec);

              const auto reshape_op_name = "System-AutoGrad-" + name + "-Reshape";
              const auto reshape_op = user_op::UserOpConfWrapperBuilder(reshape_op_name)
                                          .Op("reshape")
                                          .Input("in", scale_bn)
                                          .Attr("shape", broadcast_shape)
                                          .Output("out")
                                          .Build();
              AddOp(reshape_op);
              const auto mul_op_name = "System-AutoGrad-" + name + "-BroadcastMul";
              const auto mul_op = user_op::UserOpConfWrapperBuilder(mul_op_name)
                                      .Op("broadcast_mul")
                                      .Input("x", reshape_op.output("out", 0))
                                      .Input("y", input_bn)
                                      .Output("z")
                                      .Build();
              AddOp(mul_op);
              return mul_op;
            };
            bool fp16 = op.TensorDesc4ArgNameAndIndex("y", 0).data_type() == DataType::kFloat16;
            std::string dy_fp32_or_fp64;
            if (fp16) {
              const DataType param_data_type =
                  op.TensorDesc4ArgNameAndIndex("gamma", 0).data_type();
              const auto cast_op_name = "System-AutoGrad-" + op.op_name() + "-Cast-dy-h2f";
              const auto cast_op = user_op::UserOpConfWrapperBuilder(cast_op_name)
                                       .Op("cast")
                                       .Input("in", op.GetGradTensorWithOpOutput("y", 0))
                                       .Output("out")
                                       .Attr("dtype", param_data_type)
                                       .Build();
              AddOp(cast_op);
              dy_fp32_or_fp64 = cast_op.output("out", 0);
            } else {
              dy_fp32_or_fp64 = op.GetGradTensorWithOpOutput("y", 0);
            }
            const auto dy_mul_gamma_op =
                BroadcastMulAtAxis(op.input("gamma", 0), dy_fp32_or_fp64, "out_grad_mul_gamma");
            const auto dy_mul_inv_var_op = BroadcastMulAtAxis(
                var_sqrt_op.output("y", 0), dy_mul_gamma_op.output("z", 0), "out_grad_mul_inv_var");

            std::string dx_fp32_or_fp64 = dy_mul_inv_var_op.output("z", 0);
            std::string dx;
            if (fp16) {
              const auto cast_op_name = "System-AutoGrad-" + op.op_name() + "-Cast-dx-f2h";
              const auto cast_op = user_op::UserOpConfWrapperBuilder(cast_op_name)
                                       .Op("cast")
                                       .Input("in", dx_fp32_or_fp64)
                                       .Output("out")
                                       .Attr("dtype", DataType::kFloat16)
                                       .Build();
              AddOp(cast_op);
              dx = cast_op.output("out", 0);
            } else {
              dx = dx_fp32_or_fp64;
            }
            op.BindGradTensorWithOpInput(dx, "x", 0);
          }
        }

        const user_op::UserOpConfWrapper grad_op = grad_op_builder.Build();
        bool need_norm_grad_op = false;
        if (training && op.NeedGenGradTensor4OpInput("x", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
          need_norm_grad_op = true;
        }
        if (op.NeedGenGradTensor4OpInput("gamma", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("gamma_diff", 0), "gamma", 0);
          need_norm_grad_op = true;
        }
        if (op.NeedGenGradTensor4OpInput("beta", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("beta_diff", 0), "beta", 0);
          need_norm_grad_op = true;
        }
        if (need_norm_grad_op) { AddOp(grad_op); }
      }
    });

}  // namespace oneflow
