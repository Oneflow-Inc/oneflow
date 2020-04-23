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
    .OptionalInput("beta")
    .OptionalInput("gamma")
    .OptionalInput("cudnn_bn_scale_ones")
    .OptionalInput("cudnn_bn_bias_zeros")
    .Output("y")
    .OptionalOutput("normalized")
    .OptionalOutput("mean")
    .OptionalOutput("inv_variance")
    .Attr("alpha", UserOpAttrType::kAtFloat)
    .Attr("center", UserOpAttrType::kAtBool)
    .Attr("scale", UserOpAttrType::kAtBool)
    .Attr("begin_norm_axis", UserOpAttrType::kAtInt64)
    .Attr("begin_params_axis", UserOpAttrType::kAtInt64)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      *y = *x;

      const int64_t& begin_params_axis = ctx->GetAttr<int64_t>("begin_params_axis");
      const int64_t begin_params_axis =
          ShiftNegativeAxisIfNeed(x->shape(), conf.begin_params_axis());
      DimVector param_shape_dim_vec;
      param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                                 in->shape().dim_vec().cbegin() + begin_params_axis,
                                 in->shape().dim_vec().cend());
      if (param_shape_dim_vec.empty()) { param_shape_dim_vec.push_back(1); }
      const Shape param_shape(param_shape_dim_vec);
      if (conf.center()) {
        CHECK_OR_RETURN(
            parallel_ctx->parallel_num() == 1
            || sbp_signature->bn_in_op2sbp_parallel().at("beta").has_broadcast_parallel());
        const BlobDesc* beta = GetBlobDesc4BnInOp("beta");
        CHECK_EQ_OR_RETURN(beta->shape(), param_shape);
        CHECK_EQ_OR_RETURN(beta->data_type(), in->data_type());
      }
      if (conf.scale()) {
        CHECK_OR_RETURN(
            parallel_ctx->parallel_num() == 1
            || sbp_signature->bn_in_op2sbp_parallel().at("gamma").has_broadcast_parallel());
        const BlobDesc* gamma = GetBlobDesc4BnInOp("gamma");
        CHECK_EQ_OR_RETURN(gamma->shape(), param_shape);
        CHECK_EQ_OR_RETURN(gamma->data_type(), in->data_type());
        *GetBlobDesc4BnInOp("normalized") = *in;
      }
      const int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_norm_axis());
      DimVector bn_param_shape_dim_vec;
      bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), in->shape().dim_vec().cbegin(),
                                    in->shape().dim_vec().cbegin() + begin_norm_axis);
      const Shape bn_param_shape(bn_param_shape_dim_vec);
      BlobDesc* cudnn_bn_mean = GetBlobDesc4BnInOp("mean");
      cudnn_bn_mean->mut_shape() = bn_param_shape;
      DataType data_type =
          in->data_type() == DataType::kFloat16 ? DataType::kFloat : in->data_type();
      cudnn_bn_mean->set_data_type(data_type);
      *GetBlobDesc4BnInOp("inv_variance") = *cudnn_bn_mean;
      *GetBlobDesc4BnInOp("cudnn_bn_scale_ones") = *cudnn_bn_mean;
      *GetBlobDesc4BnInOp("cudnn_bn_bias_zeros") = *cudnn_bn_mean;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("y", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("layer_norm_grad")
    .Input("x")
    .Output("y")
    .Attr("alpha", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      *y_shape = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("y", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("layer_norm")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("layer_norm_grad")
                                                 .Input("x", op.input("x", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Output("dx")
                                                 .Attr("alpha", op.attr<float>("alpha"))
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
