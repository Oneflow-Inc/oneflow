#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("gather")
    .Input("in")
    .Input("indices")
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt64)
    .Attr("batch_dims", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      const int64_t axis = ctx->GetAttr<int64_t>("axis");
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      *out_shape = *in_shape;
      DimVector dim_vec;
      dim_vec.insert(dim_vec.end(), in_shape->dim_vec().cbegin(), in_shape->dim_vec().cbegin() + axis);
      dim_vec.insert(dim_vec.end(), indices_shape->dim_vec().cbegin(), indices_shape->dim_vec().cend());
      dim_vec.insert(dim_vec.end(), in_shape->dim_vec().cbegin() + axis + 1, in_shape->dim_vec().end());
      *out_shape = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt64;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int64_t in_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
      const int64_t indices_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();
      const int64_t gather_axis = ctx->GetAttr<int64_t>("axis");
      FOR_RANGE(int64_t, i, 0, indices_num_axes) {
        SbpSignatureBuilder()
            .Split("indices", i)
            .Broadcast("in", i)
            .Split("out", gather_axis + i)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      FOR_RANGE(int64_t, i, 0, in_num_axes) {
      if (i == gather_axis) {continue;}
      SbpSignatureBuilder()
          .Broadcast("indices")
          .Split("in", i)
          .Split("out", i < gather_axis? i : i + indices_num_axes - 1)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("gather").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  bool need_grad_in = op.NeedGenGradTensor4OpInput("in", 0);
  if (need_grad_in) {
      user_op::UserOpConfWrapperBuilder in_grad_builder(op.op_name() + "_grad");
      user_op::UserOpConfWrapper in_grad_op = in_grad_builder.Op("unsorted_segment_sum")
                                                 .Input("data", op.GetGradTensorWithOpOutput("out", 0))
                                                 .Input("segment_ids", op.input("indices", 0))
                                                 .Output("out")
                                                 .Attr("axis", op.attr<int64_t>("axis"))
                                                 .Attr("num_segments", op.attr<int64_t>("batch_dims"))
                                                 .Build();
      op.BindGradTensorWithOpInput(in_grad_op.output("out", 0), "in", 0);
      AddOp(in_grad_op);
    }
});

}  // namespace oneflow
