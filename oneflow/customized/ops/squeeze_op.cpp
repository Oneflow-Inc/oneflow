#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> TransformNegativeAxesToPositive(std::vector<int32_t>* axes, const int32_t num_axes) {
  for (auto& axis : *axes) {
    CHECK_GE(axis, -num_axes);
    CHECK_LT(axis, num_axes);
    axis = axis < 0 ? axis + num_axes : axis;
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckAndLabelAxesToSqueezeMinusOne(const std::vector<int32_t>& axes,
                                               DimVector* dim_vec) {
  for (auto axis : axes) {
    CHECK_EQ_OR_RETURN(dim_vec->at(axis), 1);
    dim_vec->at(axis) = -1;
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("squeeze")
    .Input("in")
    .Output("out")
    .Attr("axes", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      auto axes = ctx->GetAttr<std::vector<int32_t>>("axes");
      TransformNegativeAxesToPositive(&axes, in_shape->NumAxes());

      auto dim_vec = in_shape->dim_vec();
      CheckAndLabelAxesToSqueezeMinusOne(axes, &dim_vec);
      dim_vec.erase(std::remove(dim_vec.begin(), dim_vec.end(), -1), dim_vec.end());
      if (dim_vec.empty()) {
        *out_shape = Shape({1});
      } else {
        *out_shape = Shape(dim_vec);
      }
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      const auto& in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const auto* in_batch_axis = ctx->BatchAxis4ArgNameAndIndex("in", 0);
      auto* out_batch_axis = ctx->BatchAxis4ArgNameAndIndex("out", 0);
      auto axes = ctx->GetAttr<std::vector<int32_t>>("axes");
      TransformNegativeAxesToPositive(&axes, in_desc.shape().NumAxes());

      const int32_t in_batch_axis_value = static_cast<int32_t>(in_batch_axis->value());
      if (in_batch_axis->has_value()
          && std::find(axes.begin(), axes.end(), in_batch_axis_value) == axes.end()) {
        auto dim_vec = in_desc.shape().dim_vec();
        CheckAndLabelAxesToSqueezeMinusOne(axes, &dim_vec);
        const int32_t cnt = std::count(dim_vec.begin(), dim_vec.end(), -1);
        out_batch_axis->set_value(in_batch_axis_value - cnt);
      } else {
        out_batch_axis->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      auto axes = ctx->GetAttr<std::vector<int32_t>>("axes");
      TransformNegativeAxesToPositive(&axes, in_desc.shape().NumAxes());

      auto dim_vec = in_desc.shape().dim_vec();
      CheckAndLabelAxesToSqueezeMinusOne(axes, &dim_vec);
      int32_t out_axis = 0;
      FOR_RANGE(int32_t, in_axis, 0, dim_vec.size()) {
        if (in_axis != -1) {
          SbpSignatureBuilder()
              .Split("in", in_axis)
              .Split("out", out_axis)
              .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
          ++out_axis;
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("squeeze").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("reshape_like")
                                             .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                                             .Input("like", op.input("in", 0))
                                             .Output("out")
                                             .Build();
    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
    AddOp(grad_op);
  }
});

}  // namespace oneflow
