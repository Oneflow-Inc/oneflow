#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("unsorted_segment_sum")
    .Input("data")
    .Input("segment_ids")
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt64)
    .Attr("num_segments", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* data_shape = ctx->Shape4ArgNameAndIndex("data", 0);
      const int64_t axis = ctx->GetAttr<int64_t>("axis");
      const int64_t num_segments = ctx->GetAttr<int64_t>("num_segments");
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape* segment_ids_shape = ctx->Shape4ArgNameAndIndex("segment_ids", 0);
      *out_shape = *data_shape;
      DimVector dim_vec;
      dim_vec.insert(dim_vec.end(), data_shape->dim_vec().cbegin(), data_shape->dim_vec().cbegin() + axis);
      dim_vec.push_back(num_segments);
      dim_vec.insert(dim_vec.end(), data_shape->dim_vec().cbegin() + axis + segment_ids_shape->NumAxes(), data_shape->dim_vec().end());
      *out_shape = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("data", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int64_t data_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("data", 0).shape().NumAxes();
      const int64_t segment_ids_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("segment_ids", 0).shape().NumAxes();
      const int64_t axis = ctx->GetAttr<int64_t>("axis");
      FOR_RANGE(int64_t, i, 0, segment_ids_num_axes) {
      SbpSignatureBuilder()
          .Split("segment_ids", i)
          .Split("data", i + axis)
          .PartialSum("out")
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
       }
     FOR_RANGE(int64_t, i, 0, data_num_axes) {
         if (i >= axis && i < axis + segment_ids_num_axes) { continue; }
         const int64_t out_split_axis = (i < axis) ? i : i - segment_ids_num_axes + 1;
         if (out_split_axis == axis) { continue; }
         SbpSignatureBuilder()
             .Broadcast("segment_ids")
             .Split("data", i)
             .Split("out", out_split_axis)
             .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
       }
       SbpSignatureBuilder()
           .Broadcast("segment_ids")
           .PartialSum("data")
           .PartialSum("out")
           .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
           return Maybe<void>::Ok();
     });

REGISTER_USER_OP_GRAD("unsorted_segment_sum").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  bool need_grad_data = op.NeedGenGradTensor4OpInput("data", 0);
  if (need_grad_data) {
      user_op::UserOpConfWrapperBuilder data_grad_builder(op.op_name() + "_grad");
      user_op::UserOpConfWrapper data_grad_op = data_grad_builder.Op("gather")
                                                 .Input("in", op.GetGradTensorWithOpOutput("data", 0))
                                                 .Input("indices", op.input("segment_ids", 0))
                                                 .Output("out")
                                                 .Attr("axis", op.attr<int64_t>("axis"))
                                                 .Attr("batch_dims", op.attr<int64_t>("num_segments"))
                                                 .Build();
      op.BindGradTensorWithOpInput(data_grad_op.output("out", 0), "data", 0);
      AddOp(data_grad_op);
    }
});

REGISTER_USER_OP("unsorted_segment_sum_like")
    .Input("data")
    .Input("segment_ids")
    .Input("like")
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* data_shape = ctx->Shape4ArgNameAndIndex("data", 0);
      const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      const Shape* segment_ids_shape = ctx->Shape4ArgNameAndIndex("segment_ids", 0);
      const int64_t axis = ctx->GetAttr<int64_t>("axis");
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      //CHECK_EQ_OR_RETURN(data->data_type(), like->data_type());
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LE_OR_RETURN(axis, like_shape->NumAxes());
      FOR_RANGE(int64_t, i, 0, axis) {
          CHECK_EQ_OR_RETURN(like_shape->At(i), data_shape->At(i));
      }
      CHECK_EQ_OR_RETURN(data_shape->NumAxes() - segment_ids_shape->NumAxes() + 1,
                     like_shape->NumAxes());
      FOR_RANGE(int64_t, i, axis + 1, like_shape->NumAxes()) {
      CHECK_EQ_OR_RETURN(like_shape->At(i),
                       data_shape->At(i + segment_ids_shape->NumAxes() - 1));
      }
 
      //*out_shape = Shape(out_dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int64_t data_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("data", 0).shape().NumAxes();
      const int64_t segment_ids_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("segment_ids", 0).shape().NumAxes();
      const int64_t axis = ctx->GetAttr<int64_t>("axis");
      FOR_RANGE(int64_t, i, 0, segment_ids_num_axes) {
        SbpSignatureBuilder()
            .Split("segment_ids", i)
            .Split("data", i + axis)
            .Broadcast("like")
            .PartialSum("out")
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        SbpSignatureBuilder()
            .Split("segment_ids", i)
            .Split("data", i + axis)
            .PartialSum("like")
            .PartialSum("out")
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      FOR_RANGE(int64_t, i, 0, data_num_axes) {
        if (i >= axis && i < axis + segment_ids_num_axes) { continue; }
        const int64_t out_split_axis = (i < axis) ? i : i - segment_ids_num_axes + 1;
        if (out_split_axis == axis) { continue; }
        SbpSignatureBuilder()
            .Broadcast("segment_ids")
            .Split("data", i)
            .Split("like", out_split_axis)
            .Split("out", out_split_axis)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      SbpSignatureBuilder()
          .Broadcast("segment_ids")
          .PartialSum("data")
          .Broadcast("like")
          .PartialSum("out")
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder()
          .Broadcast("segment_ids")
          .PartialSum("data")
          .PartialSum("like")
          .PartialSum("out")
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
         return Maybe<void>::Ok();
         });

}  // namespace oneflow
