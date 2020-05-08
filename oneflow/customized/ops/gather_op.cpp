#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("gather")
    .Input("in")
    .Input("indices")
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      CHECK_GT_OR_RETURN(in_shape->NumAxes(), 0);
      const int64_t axis = ctx->GetAttr<int64_t>("axis");
      const user_op::TensorDesc* indices = ctx->TensorDesc4ArgNameAndIndex("indices", 0);
      const Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      CHECK_OR_RETURN(IsIndexDataType(indices->data_type()));
      CHECK_GT_OR_RETURN(indices_shape->NumAxes(), 0);
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);

      *out_shape = *in_shape;
      DimVector dim_vec;
      dim_vec.insert(dim_vec.end(), in_shape->dim_vec().cbegin(),
                     in_shape->dim_vec().cbegin() + axis);
      dim_vec.insert(dim_vec.end(), indices_shape->dim_vec().cbegin(),
                     indices_shape->dim_vec().cend());
      dim_vec.insert(dim_vec.end(), in_shape->dim_vec().cbegin() + axis + 1,
                     in_shape->dim_vec().end());
      *out_shape = Shape(dim_vec);
      out->set_is_dynamic(indices->is_dynamic());
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      if (ctx->BatchAxis4ArgNameAndIndex("indices", 0)->has_value()) {
        ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(
            ctx->GetAttr<int64_t>("axis") + ctx->BatchAxis4ArgNameAndIndex("indices", 0)->value());
      } else {
        ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int64_t in_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
      const int64_t indices_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();
      const int64_t gather_axis = ctx->GetAttr<int64_t>("axis");
      CHECK_GE_OR_RETURN(gather_axis, 0);
      CHECK_LT_OR_RETURN(gather_axis, in_num_axes);
      FOR_RANGE(int64_t, i, 0, indices_num_axes) {
        SbpSignatureBuilder()
            .Split("indices", 0, i)
            .Broadcast("in", 0)
            .Split("out", 0, gather_axis + i)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      FOR_RANGE(int64_t, i, 0, in_num_axes) {
        if (i == gather_axis) { continue; }
        SbpSignatureBuilder()
            .Broadcast("indices", 0)
            .Split("in", 0, i)
            .Split("out", 0, i < gather_axis ? i : i + indices_num_axes - 1)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("gather").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  bool need_grad_in = op.NeedGenGradTensor4OpInput("in", 0);
  if (need_grad_in) {
    user_op::UserOpConfWrapperBuilder in_grad_builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper in_grad_op =
        in_grad_builder.Op("unsorted_segment_sum_like")
            .Input("data", op.GetGradTensorWithOpOutput("out", 0))
            .Input("segment_ids", op.input("indices", 0))
            .Input("like", op.input("in", 0))
            .Output("out")
            .Attr("axis", op.attr<int64_t>("axis"))
            .Build();
    op.BindGradTensorWithOpInput(in_grad_op.output("out", 0), "in", 0);
    AddOp(in_grad_op);
  }
});

}  // namespace oneflow
