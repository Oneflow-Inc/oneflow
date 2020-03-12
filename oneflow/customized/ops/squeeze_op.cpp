#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> LabelAxesToSqueezeMinusOne(const std::vector<int32_t>& axes, DimVector* dim_vec) {
  for (auto axis : axes) {
    CHECK_GE_OR_RETURN(axis, 0);
    CHECK_LT_OR_RETURN(axis, dim_vec->size());
    dim_vec->at(axis) = -1;
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("squeeze")
    .Input("in")
    .Output("out")
    .Attr("axes", UserOpAttrType::kAtListInt32)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      auto axes = ctx->GetAttr<std::vector<int32_t>>("axes");

      auto dim_vec = in_shape->dim_vec();
      // TODO (xfjiang): deal with empty blob
      for (auto dim : dim_vec) { CHECK_GT_OR_RETURN(dim, 0); }
      // Just remove the axis user want to squeeze at compile stage and will check
      // "in_shape->At(axis) == 1" at runtime stage with dynamic shape
      LabelAxesToSqueezeMinusOne(axes, &dim_vec);
      dim_vec.erase(std::remove(dim_vec.begin(), dim_vec.end(), -1), dim_vec.end());
      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      const auto* in_batch_axis = ctx->BatchAxis4ArgNameAndIndex("in", 0);
      auto* out_batch_axis = ctx->BatchAxis4ArgNameAndIndex("out", 0);
      const auto axes = ctx->GetAttr<std::vector<int32_t>>("axes");

      if (in_batch_axis->has_value()
          && std::find(axes.begin(), axes.end(), static_cast<int32_t>(in_batch_axis->value()))
                 == axes.end()) {
        auto dim_vec = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().dim_vec();
        LabelAxesToSqueezeMinusOne(axes, &dim_vec);
        int32_t cnt = 0;
        FOR_RANGE(int32_t, i, 0, static_cast<int32_t>(in_batch_axis->value())) {
          cnt += static_cast<int32_t>(dim_vec.at(i) == -1);
        }
        out_batch_axis->set_value(in_batch_axis->value() - cnt);
      } else {
        out_batch_axis->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const auto axes = ctx->GetAttr<std::vector<int32_t>>("axes");

      auto dim_vec = in_desc.shape().dim_vec();
      LabelAxesToSqueezeMinusOne(axes, &dim_vec);
      int32_t out_axis = 0;
      FOR_RANGE(int32_t, in_axis, 0, dim_vec.size()) {
        if (in_axis != -1) {
          SbpSignatureBuilder()
              .Split("in", in_axis)
              .Split("out", out_axis++)
              .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("squeeze").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op =
        builder.Op("reshape_like")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Input("like", op.input("in", 0))
            .Output("out")
            .Build();
    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
    AddOp(grad_op);
  }
});

}  // namespace oneflow
