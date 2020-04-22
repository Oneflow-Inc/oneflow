#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("normalization")
    .Input("in")
    .Input("moving_mean")
    .Input("moving_variance")
    // TODO: python 判断这里要不要传 const op
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
      // TODO: 这是必要的吗
      // (conf.has_gamma() ? CheckParamBlobDesc : SetParamBlobDesc)("gamma");
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
      // TODO: 确认这里对不对
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
            std::cout << __FILE__ << " " << __LINE__ << op.NeedGenGradTensor4OpInput("in", 0) << std::endl;
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("normalization_grad")
                                                 .Input("x", op.input("in", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                                                 // TODO: use moving_mean when not training
                                                 .Input("mean", op.output("mean", 0))
                                                 .Input("inv_variance", op.output("inv_variance", 0))
                                                 .Input("gamma", op.input("gamma", 0))
                                                 .Attr("axis", op.attr<int32_t>("axis"))
                                                 .Attr("epsilon", op.attr<float>("epsilon"))
                                                 .Output("gamma_diff")
                                                 .Output("beta_diff")
                                                 .Output("dx")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "in", 0);
        op.BindGradTensorWithOpInput(grad_op.output("gamma_diff", 0), "gamma", 0);
        op.BindGradTensorWithOpInput(grad_op.output("beta_diff", 0), "beta", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
