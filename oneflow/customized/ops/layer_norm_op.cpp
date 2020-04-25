#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

int64_t ShiftNegativeAxisIfNeed(const Shape& shape, int64_t axis) {
  const int64_t shifted = axis < 0 ? axis + shape.NumAxes() : axis;
  CHECK_GE(shifted, 0);
  CHECK_LT(shifted, shape.NumAxes());
  return shifted;
}

}  // namespace

REGISTER_USER_OP("layer_norm")
    .Input("x")
    .Output("y")
    .Output("mean")
    .Output("inv_variance")
    .OptionalInput("beta")
    .OptionalInput("gamma")
    .Input("cudnn_bn_scale_ones")
    .Input("cudnn_bn_bias_zeros")
    .OptionalOutput("normalized")
    .Attr("center", UserOpAttrType::kAtBool)
    .Attr("scale", UserOpAttrType::kAtBool)
    .Attr("begin_norm_axis", UserOpAttrType::kAtInt64)
    .Attr("begin_params_axis", UserOpAttrType::kAtInt64)
    .Attr("epsilon", UserOpAttrType::kAtDouble)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      const user_op::TensorDesc* beta = ctx->TensorDesc4ArgNameAndIndex("beta", 0);
      const user_op::TensorDesc* gamma = ctx->TensorDesc4ArgNameAndIndex("gamma", 0);
      const user_op::TensorDesc* cudnn_bn_scale_ones =
          ctx->TensorDesc4ArgNameAndIndex("cudnn_bn_scale_ones", 0);
      const user_op::TensorDesc* cudnn_bn_bias_zeros =
          ctx->TensorDesc4ArgNameAndIndex("cudnn_bn_bias_zeros", 0);
      user_op::TensorDesc* normalized = ctx->TensorDesc4ArgNameAndIndex("normalized", 0);
      user_op::TensorDesc* mean = ctx->TensorDesc4ArgNameAndIndex("mean", 0);
      user_op::TensorDesc* inv_variance = ctx->TensorDesc4ArgNameAndIndex("inv_variance", 0);
      const bool& center = ctx->GetAttr<bool>("center");
      const bool& scale = ctx->GetAttr<bool>("scale");
      const int64_t begin_params_axis =
          ShiftNegativeAxisIfNeed(x->shape(), ctx->GetAttr<int64_t>("begin_params_axis"));
      *y = *x;
      DimVector param_shape_dim_vec;
      param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                                 x->shape().dim_vec().cbegin() + begin_params_axis,
                                 x->shape().dim_vec().cend());
      if (param_shape_dim_vec.empty()) { param_shape_dim_vec.push_back(1); }
      const Shape param_shape(param_shape_dim_vec);
      if (center) {
        CHECK_OR_RETURN(ctx->parallel_ctx().parallel_num() == 1
                        || ctx->SbpParallel4ArgNameAndIndex("beta", 0).has_broadcast_parallel());
        CHECK_EQ_OR_RETURN(beta->shape(), param_shape);
        CHECK_EQ_OR_RETURN(beta->data_type(), x->data_type());
      }
      if (scale) {
        CHECK_OR_RETURN(ctx->parallel_ctx().parallel_num() == 1
                        || ctx->SbpParallel4ArgNameAndIndex("gamma", 0).has_broadcast_parallel());
        CHECK_EQ_OR_RETURN(gamma->shape(), param_shape);
        CHECK_EQ_OR_RETURN(gamma->data_type(), x->data_type());
        *normalized = *x;
      }
      const int64_t begin_norm_axis =
          ShiftNegativeAxisIfNeed(x->shape(), ctx->GetAttr<int64_t>("begin_norm_axis"));
      DimVector bn_param_shape_dim_vec;
      bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), x->shape().dim_vec().cbegin(),
                                    x->shape().dim_vec().cbegin() + begin_norm_axis);
      const Shape bn_param_shape(bn_param_shape_dim_vec);
      *mean->mut_shape() = bn_param_shape;
      DataType data_type = x->data_type() == DataType::kFloat16 ? DataType::kFloat : x->data_type();
      *mean->mut_data_type() = data_type;
      *inv_variance = *mean;
      CHECK_EQ_OR_RETURN(cudnn_bn_scale_ones->shape(), mean->shape());
      CHECK_EQ_OR_RETURN(cudnn_bn_scale_ones->data_type(), mean->data_type());
      CHECK_EQ_OR_RETURN(cudnn_bn_bias_zeros->shape(), mean->shape());
      CHECK_EQ_OR_RETURN(cudnn_bn_bias_zeros->data_type(), mean->data_type());
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      for (const auto& ob : ctx->outputs()) {
        *ctx->BatchAxis4ArgNameAndIndex(ob.first, ob.second) =
            *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Broadcast("gamma", 0)
          .Broadcast("beta", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("layer_norm_grad")
    .Input("dy")
    .Input("x")
    .Input("cudnn_bn_scale_ones")
    .OptionalInput("mean")
    .OptionalInput("inv_variance")
    .Output("dx")
    .Output("cudnn_bn_scale_diff_buf")
    .Output("cudnn_bn_bias_diff_buf")
    .Attr("begin_norm_axis", UserOpAttrType::kAtInt64)
    .Attr("epsilon", UserOpAttrType::kAtDouble)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* mean = ctx->TensorDesc4ArgNameAndIndex("mean", 0);
      const user_op::TensorDesc* inv_variance = ctx->TensorDesc4ArgNameAndIndex("inv_variance", 0);
      const user_op::TensorDesc* cudnn_bn_scale_ones =
          ctx->TensorDesc4ArgNameAndIndex("cudnn_bn_scale_ones", 0);
      const user_op::TensorDesc* cudnn_bn_scale_diff_buf =
          ctx->TensorDesc4ArgNameAndIndex("cudnn_bn_scale_diff_buf", 0);
      const user_op::TensorDesc* cudnn_bn_bias_diff_buf =
          ctx->TensorDesc4ArgNameAndIndex("cudnn_bn_bias_diff_buf", 0);
      user_op::TensorDesc* dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      int64_t begin_norm_axis = ctx->GetAttr<int64_t>("begin_norm_axis");
      CHECK_GE_OR_RETURN(begin_norm_axis, 1);
      CHECK_LT_OR_RETURN(begin_norm_axis, dy->shape().NumAxes());
      CHECK_EQ_OR_RETURN(dy->data_type(), x->data_type());
      CHECK_EQ_OR_RETURN(dy->shape(), x->shape());
      begin_norm_axis =
          begin_norm_axis < 0 ? dy->shape().NumAxes() + begin_norm_axis : begin_norm_axis;
      CHECK_GE_OR_RETURN(begin_norm_axis, 1);
      CHECK_LT_OR_RETURN(begin_norm_axis, x->shape().NumAxes());
      DimVector bn_param_shape_dim_vec;
      bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), x->shape().dim_vec().cbegin(),
                                    x->shape().dim_vec().cbegin() + begin_norm_axis);
      const Shape bn_param_shape(bn_param_shape_dim_vec);
      if (mean || inv_variance) {
        CHECK_OR_RETURN(mean);
        CHECK_OR_RETURN(inv_variance);
        // CHECK_EQ(mean->data_type(), x->data_type());
        CHECK_EQ_OR_RETURN(mean->shape(), bn_param_shape);
        // CHECK_EQ(inv_variance->data_type(), x->data_type());
        CHECK_EQ_OR_RETURN(inv_variance->shape(), bn_param_shape);
      }
      *dx = *dy;
      CHECK_EQ_OR_RETURN(cudnn_bn_scale_ones->shape(),
                         Shape({dy->shape().Count(0, begin_norm_axis)}));
      DataType data_type =
          dy->data_type() == DataType::kFloat16 ? DataType::kFloat : dy->data_type();
      CHECK_EQ_OR_RETURN(cudnn_bn_scale_ones->data_type(), data_type);
      CHECK_EQ_OR_RETURN(cudnn_bn_scale_diff_buf->shape(), cudnn_bn_scale_ones->shape());
      CHECK_EQ_OR_RETURN(cudnn_bn_scale_diff_buf->data_type(), cudnn_bn_scale_ones->data_type());
      CHECK_EQ_OR_RETURN(cudnn_bn_bias_diff_buf->shape(), cudnn_bn_scale_ones->shape());
      CHECK_EQ_OR_RETURN(cudnn_bn_bias_diff_buf->data_type(), cudnn_bn_scale_ones->data_type());
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      for (const auto& ob : ctx->outputs()) {
        *ctx->BatchAxis4ArgNameAndIndex(ob.first, ob.second) =
            *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("layer_norm_param_grad")
    .Input("dy")
    .OptionalInput("normalized")
    .OptionalInput("gamma")
    .OptionalOutput("normalized_diff")
    .OptionalOutput("beta_diff")
    .OptionalOutput("gamma_diff")
    .OptionalOutput("reduce_buf")
    .Attr("begin_params_axis", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const user_op::TensorDesc* normalized = ctx->TensorDesc4ArgNameAndIndex("normalized", 0);
      const user_op::TensorDesc* gamma = ctx->TensorDesc4ArgNameAndIndex("gamma", 0);
      const user_op::TensorDesc* reduce_buf = ctx->TensorDesc4ArgNameAndIndex("reduce_buf", 0);
      user_op::TensorDesc* normalized_diff = ctx->TensorDesc4ArgNameAndIndex("normalized_diff", 0);
      user_op::TensorDesc* beta_diff = ctx->TensorDesc4ArgNameAndIndex("beta_diff", 0);
      user_op::TensorDesc* gamma_diff = ctx->TensorDesc4ArgNameAndIndex("gamma_diff", 0);
      int64_t begin_params_axis = ctx->GetAttr<int64_t>("begin_params_axis");
      const bool& has_beta_diff = beta_diff != nullptr;
      const bool& has_gamma_diff = gamma_diff != nullptr;
      const bool& has_gamma = gamma != nullptr;
      const bool& has_normalized_diff = normalized_diff != nullptr;
      if (has_beta_diff || has_gamma_diff) {
        CHECK_EQ_OR_RETURN(reduce_buf->data_type(), dy->data_type());
        CHECK_EQ_OR_RETURN(reduce_buf->shape(), dy->shape());
      }
      begin_params_axis =
          begin_params_axis < 0 ? dy->shape().NumAxes() + begin_params_axis : begin_params_axis;
      CHECK_GE_OR_RETURN(begin_params_axis, 1);
      CHECK_LT_OR_RETURN(begin_params_axis, dy->shape().NumAxes());
      DimVector param_shape_dim_vec;
      param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                                 dy->shape().dim_vec().cbegin() + begin_params_axis,
                                 dy->shape().dim_vec().cend());
      if (param_shape_dim_vec.empty()) { param_shape_dim_vec.push_back(1); }
      const Shape param_shape(param_shape_dim_vec);
      if (has_beta_diff) {
        CHECK_EQ_OR_RETURN(beta_diff->data_type(), beta_diff->data_type());
        CHECK_EQ_OR_RETURN(beta_diff->shape(), beta_diff->shape());
      }
      if (has_gamma_diff) {
        CHECK_EQ_OR_RETURN(normalized->data_type(), normalized->data_type());
        CHECK_EQ_OR_RETURN(normalized->shape(), normalized->shape());
        CHECK_EQ_OR_RETURN(gamma_diff->data_type(), gamma_diff->data_type());
        CHECK_EQ_OR_RETURN(gamma_diff->shape(), gamma_diff->shape());
      }
      if (has_normalized_diff) { *normalized_diff = *dy; }
      if (has_gamma) {
        CHECK_OR_RETURN(ctx->parallel_ctx().parallel_num() == 1
                        || ctx->SbpParallel4ArgNameAndIndex("gamma", 0).has_broadcast_parallel());
        CHECK_EQ_OR_RETURN(gamma->data_type(), dy->data_type());
        CHECK_EQ_OR_RETURN(gamma->shape(), param_shape);
      }
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      for (const auto& ob : ctx->outputs()) {
        ctx->BatchAxis4ArgNameAndIndex(ob.first, ob.second)->clear_value();
      }
      // TODO: tsai: normalized_diff could be null
      *ctx->BatchAxis4ArgNameAndIndex("normalized_diff", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Broadcast("gamma")
          .Split(ctx->outputs(), 0)
          .Broadcast("gamma_diff", 0)
          .Broadcast("beta_diff", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("layer_norm")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      const bool has_beta_diff = op.NeedGenGradTensor4OpInput("beta", 0);
      const bool has_gamma_diff = op.NeedGenGradTensor4OpInput("gamma", 0);
      const bool need_scale_out_diff = op.NeedGenGradTensor4OpInput("x", 0);
      const int64_t& begin_params_axis = op.attr<int64_t>("begin_params_axis");
      const int64_t num_in_axes = op.TensorDesc4ArgNameAndIndex("x", 0).shape().NumAxes();
      const auto ShiftAxisIfNeed = [num_in_axes](const int64_t axis) {
        return axis < 0 ? axis + num_in_axes : axis;
      };
      std::string dy = op.GetGradTensorWithOpOutput("y", 0);
      if (has_beta_diff || has_gamma_diff || need_scale_out_diff) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_param_grad");
        auto grad_op_builder = builder.Op("layer_norm_param_grad")
                                   .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                   .Attr("begin_params_axis", ShiftAxisIfNeed(begin_params_axis));
        if (has_beta_diff) { grad_op_builder.Output("beta_diff"); }
        if (has_gamma_diff || need_scale_out_diff) {
          grad_op_builder.Input("gamma", op.input("gamma", 0));
        }
        if (has_gamma_diff) {
          grad_op_builder.Input("normalized", op.input("normalized", 0));
          grad_op_builder.Output("gamma_diff");
        }
        if (need_scale_out_diff) { grad_op_builder.Output("normalized_diff"); }
        auto grad_op = grad_op_builder.Build();
        if (has_beta_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("beta_diff", 0), "beta", 0);
        }
        if (has_gamma_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("gamma_diff", 0), "gamma", 0);
        }
        if (need_scale_out_diff) { dy = grad_op.output("normalized_diff", 0); }
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("layer_norm_grad")
                .Input("x", op.input("x", 0))
                .Input("dy", dy)
                .Input("mean", op.output("mean", 0))
                .Input("inv_variance", op.output("inv_variance", 0))
                .Output("dx")
                .Attr("begin_params_axis", ShiftAxisIfNeed(begin_params_axis))
                .Attr("epsilon", op.attr<double>("epsilon"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
