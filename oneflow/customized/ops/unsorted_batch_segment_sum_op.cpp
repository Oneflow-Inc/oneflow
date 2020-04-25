#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("unsorted_batch_segment_sum")
    .Input("data")
    .Input("segment_ids")
    .Output("out")
    .Attr("num_segments", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* data_shape = ctx->Shape4ArgNameAndIndex("data", 0);
      const int64_t num_segments = ctx->GetAttr<int64_t>("num_segments");
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape* segment_ids_shape = ctx->Shape4ArgNameAndIndex("segment_ids", 0);
      *out_shape = *data_shape;
      CHECK_GE_OR_RETURN(segment_ids->shape().NumAxes(), 1);
      CHECK_GE_OR_RETURN(data->shape().NumAxes(), segment_ids->shape().NumAxes());
      CHECK_EQ(segment_ids->is_dynamic(), data->is_dynamic());
      FOR_RANGE(int64_t, i, 0, segment_ids->shape().NumAxes() - 1) {
          CHECK_EQ_OR_RETURN(segment_ids->shape().At(i), data->shape().At(i));
      }

      DimVector dim_vec(data_shape->dim_vec());
      dim_vec->at(segment_ids_shape).NumAxes() - 1) = num_segments;
      *out_shape = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("data", 0);
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
    FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
    SbpSignatureBuilder()
        .Split("segment_ids", i)
        .Split("data", i)
        .Split("out", i)
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
      user_op::UserOpConfWrapper data_grad_op = data_grad_builder.Op("batch_gather")
                                                 .Input("in", op.GetGradTensorWithOpOutput("data", 0))
                                                 .Input("indices", op.input("segment_ids", 0))
                                                 .Output("out")
                                                 .Build();
      op.BindGradTensorWithOpInput(data_grad_op.output("out", 0), "data", 0);
      AddOp(data_grad_op);
    }
});

}  // namespace oneflow
