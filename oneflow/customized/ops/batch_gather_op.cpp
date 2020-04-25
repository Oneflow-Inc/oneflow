#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("batch_gather")
    .Input("in")
    .Input("indices")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      const int64_t axis = ctx->GetAttr<int64_t>("axis");
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      *out_shape = *in_shape;
      DimVector dim_vec;
      dim_vec->at(indices_shape.NumAxes() - 1) = indice_shape.back();
      *out_shape = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Tensor* in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const int64_t in_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
      const int64_t indices_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();
      const int64_t gather_axis = ctx->GetAttr<int64_t>("axis");
      CHECK_LE_OR_RETURN(indices_dim_vec.size(), in_dim_vec.size());
      FOR_RANGE(int64_t, i, 0, indices_dim_vec.size() - 1) {
        if (in->is_dynamic() && indices->is_dynamic() == false) {
          CHECK_GE_OR_RETURN(indices_dim_vec.at(i), in_dim_vec.at(i));
        } else if (in->is_dynamic() == false && indices->is_dynamic()) {
          UNIMPLEMENTED();
        } else {
          CHECK_EQ_OR_RETURN(indices_dim_vec.at(i), in_dim_vec.at(i));
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("batch_gather").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  bool need_grad_in = op.NeedGenGradTensor4OpInput("in", 0);
  if (need_grad_in) {
      user_op::UserOpConfWrapperBuilder in_grad_builder(op.op_name() + "_grad");
      user_op::UserOpConfWrapper in_grad_op = in_grad_builder.Op("unsorted_batch_segment_sum")
                                                 .Input("data", op.GetGradTensorWithOpOutput("out", 0))
                                                 .Input("segment_ids", op.input("indices", 0))
                                                 .Output("out")
                                                 .Attr("num_segments", op.input("in", 0)->shape.At(op.input("indices")->shape.NumAxes() -1))
                                                 .Build();
      op.BindGradTensorWithOpInput(in_grad_op.output("out", 0), "in", 0);
      AddOp(in_grad_op);
    }
});

}  // namespace oneflow
