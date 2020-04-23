#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("normalization")
    .Input("in")
    .Input("moving_mean")
    .Input("moving_variance")
    .Input("gamma")
    .Input("beta")
    .Output("out")
    .OptionalOutput("mean")
    .OptionalOutput("inv_variance")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .Attr("is_training", UserOpAttrType::kAtBool)
    .Attr("momentum", UserOpAttrType::kAtFloat)
    .Attr("center", UserOpAttrType::kAtBool)
    .Attr("scale", UserOpAttrType::kAtBool)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* moving_mean_modifier = GetInputArgModifierFn("moving_mean", 0);
      CHECK(moving_mean_modifier != nullptr);
      // TODO: get is_training
      bool is_training = true;
      moving_mean_modifier->set_is_mutable(is_training);
      moving_mean_modifier->set_requires_grad(false);
      user_op::InputArgModifier* moving_variance_modifier =
          GetInputArgModifierFn("moving_variance", 0);
      CHECK(moving_variance_modifier != nullptr);
      moving_variance_modifier->set_is_mutable(is_training);
      moving_variance_modifier->set_requires_grad(false);
    })
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const auto* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const DataType data_type = in->data_type();
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *in;
      int32_t axis = ctx->GetAttr<int32_t>("axis");
      OF_CHECK_GE(axis, 0);
      OF_CHECK_LT(axis, in->shape().NumAxes());
      const Shape param_shape({in->shape().At(axis)});
      const std::function<void(const std::string&)> CheckParamTensorDesc =
          [&](const std::string& bn) -> Maybe<void> {
        const auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
        if (tensor_desc != nullptr) {
          CHECK_EQ_OR_RETURN(tensor_desc->data_type(), data_type);
          CHECK_EQ_OR_RETURN(tensor_desc->shape(), param_shape);
        }
        return Maybe<void>::Ok();
      };
      const std::function<void(const std::string&)> SetParamTensorDesc =
          [&](const std::string& bn) -> Maybe<void> {
        auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
        if (tensor_desc != nullptr) {
          *tensor_desc->mut_data_type() = data_type;
          *tensor_desc->mut_shape() = param_shape;
        }
        return Maybe<void>::Ok();
      };
      CheckParamTensorDesc("moving_mean");
      CheckParamTensorDesc("moving_variance");
      if (ctx->GetAttr<bool>("center")) {
        CheckParamTensorDesc("beta");
      } else {
        SetParamTensorDesc("beta");
      }
      if (ctx->GetAttr<bool>("scale")) {
        CheckParamTensorDesc("gamma");
      } else {
        SetParamTensorDesc("gamma");
      }
      if (ctx->GetAttr<bool>("is_training")) {
        SetParamTensorDesc("mean");
        SetParamTensorDesc("inv_variance");
      }
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      if (ctx->GetAttr<bool>("is_training")) {
        ctx->BatchAxis4ArgNameAndIndex("mean", 0)->clear_value();
        ctx->BatchAxis4ArgNameAndIndex("inv_variance", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);

      SbpSignatureBuilder()
          .Broadcast(ctx->inputs())
          .Broadcast(ctx->outputs())
          .Split("in", 0)
          .Split("out", 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

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
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const auto x_type = *ctx->Dtype4ArgNameAndIndex("x", 0);
      const auto dy_type = *ctx->Dtype4ArgNameAndIndex("dy", 0);
      CHECK_EQ_OR_RETURN(x_type, dy_type);
      const Shape x_shape = *ctx->Shape4ArgNameAndIndex("x", 0);
      const Shape dy_shape = *ctx->Shape4ArgNameAndIndex("dy", 0);
      CHECK_EQ_OR_RETURN(dy_shape, x_shape);
      *ctx->Shape4ArgNameAndIndex("dx", 0) = dy_shape;
      auto* dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      if (dx) { *dx = *x; }
      const Shape param_shape({x_shape.At(ctx->GetAttr<int32_t>("axis"))});
      const std::function<void(const std::string&)> CheckParamBlobDesc =
          [&](const std::string& bn) -> Maybe<void> {
        CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex(bn, 0), dy_type);
        const auto* blob_shape = ctx->Shape4ArgNameAndIndex(bn, 0);
        if (blob_shape) { CHECK_EQ_OR_RETURN(*blob_shape, param_shape); }
        return Maybe<void>::Ok();
      };
      const std::function<void(const std::string&)> SetParamBlobDesc =
          [&](const std::string& bn) -> Maybe<void> {
        auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
        if (tensor_desc) {
          *tensor_desc->mut_data_type() = dy_type;
          *tensor_desc->mut_shape() = param_shape;
        }
        return Maybe<void>::Ok();
      };

      CheckParamBlobDesc("mean");
      CheckParamBlobDesc("inv_variance");
      CheckParamBlobDesc("gamma");
      SetParamBlobDesc("gamma_diff");
      SetParamBlobDesc("beta_diff");
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("dy", 0);
      ctx->BatchAxis4ArgNameAndIndex("gamma_diff", 0)->clear_value();
      ctx->BatchAxis4ArgNameAndIndex("beta_diff", 0)->clear_value();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Broadcast(ctx->inputs())
          .PartialSum(ctx->outputs())
          .Split("x", 0)
          .Split("dx", 0)
          .Split("dy", 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("normalization")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0) || op.NeedGenGradTensor4OpInput("gamma", 0)
          || op.NeedGenGradTensor4OpInput("beta", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        auto grad_op_builder = builder.Op("normalization_grad")
                                   .Input("x", op.input("in", 0))
                                   .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                                   .Attr("axis", op.attr<int32_t>("axis"))
                                   .Attr("epsilon", op.attr<float>("epsilon"))
                                   .Output("gamma_diff")
                                   .Output("beta_diff")
                                   .Output("dx");
        bool is_training = op.attr<bool>("is_training");
        if (is_training) {
          grad_op_builder = grad_op_builder.Input("mean", op.output("mean", 0))
                                .Input("inv_variance", op.output("inv_variance", 0));
        } else {
          // this part has not been tested, blocked by reshape, broadcast_mul user op
          // and the disability of getting attr in input arg modifier
          grad_op_builder.Input("mean", op.output("moving_mean", 0));

          // calculate inv_variance from moving_variance
          const auto scalar_add_op_name = "System-AutoGrad-" + op.op_name() + "-VarianceAddEpsilon";
          auto scalar_add_op = user_op::UserOpConfWrapperBuilder(scalar_add_op_name)
                                   .Op("scalar_add")
                                   .Input("in", op.output("moving_variance", 0))
                                   .Attr("has_float_operand", true)
                                   .Attr("has_int_operand", false)
                                   .Attr("int_operand", 0)
                                   .Attr("float_operand", op.attr<float>("epsilon"))
                                   .Output("out")
                                   .Build();
          AddOp(scalar_add_op);

          const auto rsqrt_op_name = "System-AutoGrad-" + op.op_name() + "-InvVarianceRsqrt";
          auto rsqrt_op = user_op::UserOpConfWrapperBuilder(rsqrt_op_name)
                              .Op("rsqrt")
                              .Input("in", scalar_add_op.output("out", 0))
                              .Output("out")
                              .Build();
          AddOp(rsqrt_op);

          grad_op_builder.Input("inv_variance", rsqrt_op.output("out", 0));

          if (op.NeedGenGradTensor4OpInput("in", 0)) {
            // calculate dx manually as cudnn cannot be used in evaluation mode
            // reference: https://github.com/pytorch/pytorch/issues/4284
            const auto axis = op.attr<int32_t>("axis");
            auto BroadcastMulAtAxis = [&op, &axis, &AddOp](const std::string& scale_bn,
                                                           const std::string& input_bn,
                                                           const std::string& name) {
              std::vector<int64_t> broadcast_shape;
              auto in_shape = op.TensorDesc4ArgNameAndIndex("in", 0).shape();
              FOR_RANGE(size_t, i, 0, in_shape.NumAxes()) {
                if (i != axis) {
                  broadcast_shape.push_back(1);
                } else {
                  broadcast_shape.push_back(in_shape.At(axis));
                }
              }

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
                                      .Input("a", reshape_op.output("out", 0))
                                      .Input("b", input_bn)
                                      .Output("out")
                                      .Build();
              AddOp(mul_op);
              return mul_op;
            };
            const auto mul_gamma_op = BroadcastMulAtAxis(
                op.input("gamma", 0), op.GetGradTensorWithOpOutput("out", 0), "mul_gamma");
            const auto mul_inv_var_op = BroadcastMulAtAxis(
                rsqrt_op.output("out", 0), mul_gamma_op.output("out", 0), "mul_inv_var");
            op.BindGradTensorWithOpInput(mul_inv_var_op.output("out", 0), "in", 0);
          }
        }
        user_op::UserOpConfWrapper grad_op = grad_op_builder.Build();
        if (is_training && op.NeedGenGradTensor4OpInput("in", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "in", 0);
        }
        if (op.NeedGenGradTensor4OpInput("gamma", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("gamma_diff", 0), "gamma", 0);
        }
        if (op.NeedGenGradTensor4OpInput("beta", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("beta_diff", 0), "beta", 0);
        }
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
