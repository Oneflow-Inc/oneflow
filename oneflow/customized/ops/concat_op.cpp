#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("concat")
    .InputWithMinimum("in", 2)
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const int32_t axis = ctx->Attr<int32_t>("axis");
      const user_op::TensorDesc* in_0_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      DimVector out_dim_vec = in_0_desc->shape().dim_vec();

      for (size_t i = 1; i < ctx->inputs().size(); ++i) {
        const user_op::TensorDesc* in_i_desc = ctx->TensorDesc4ArgNameAndIndex("in", i);
        for (int64_t j = 0; j < in_i_desc->shape().NumAxes(); ++j) {
          if (j == axis) {
            out_dim_vec[j] += in_i_desc->shape().At(j);
          } else {
            CHECK_EQ_OR_RETURN(out_dim_vec[j], in_i_desc->shape().At(j));
          }
        }
        CHECK_EQ_OR_RETURN(in_i_desc->data_type(), in_0_desc->data_type());
      }
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *in_0_desc;
      *out_desc->mut_shape() = Shape(out_dim_vec);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int32_t axis = ctx->Attr<int32_t>("axis");
      const user_op::TensorDesc& in_0_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in_0_tensor.shape().NumAxes()) {
        if (i == axis) { continue; }
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("concat").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  bool need_grad = false;
  int32_t in_size = op.input_size("in");
  for (size_t i = 0; i < in_size; ++i) {
    if (op.NeedGenGradTensor4OpInput("in", i)) { need_grad = true; }
  }
  if (need_grad) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    for (size_t i = 0; i < in_size; ++i) { builder = builder.Input("like", op.input("in", i)); }
    user_op::UserOpConfWrapper grad_op = builder.Op("split_like")
                                             .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("out", in_size)
                                             .Attr("axis", op.attr<int32_t>("axis"))
                                             .Build();

    for (size_t i = 0; i < in_size; ++i) {
      if (op.NeedGenGradTensor4OpInput("in", i)) {
        op.BindGradTensorWithOpInput(grad_op.output("out", i), "in", i);
      }
    }
    AddOp(grad_op);
  }
});

}  // namespace oneflow
