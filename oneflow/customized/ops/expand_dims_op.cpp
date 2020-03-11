#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("expand_dims")
    .Input("in")
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      int32_t axis = ctx->GetAttr<int32_t>("axis");
      axis = axis < 0 ? axis + in_shape->NumAxes() : axis;
      CHECK_GE(axis, 0);
      CHECK_LT(axis, in_shape->NumAxes());

      auto dim_vec = in_shape->dim_vec();
      dim_vec.insert(dim_vec.begin() + axis, 1);
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
      const int32_t axis = ctx->GetAttr<int32_t>("axis");

      if (in_batch_axis->has_value()) {
        out_batch_axis->set_value(axis <= static_cast<int32_t>(in_batch_axis->value())
                                      ? in_batch_axis->value() + 1
                                      : in_batch_axis->value());
      } else {
        out_batch_axis->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const int32_t axis = ctx->GetAttr<int32_t>("axis");

      auto dim_vec = in_desc.shape().dim_vec();
      FOR_RANGE(int32_t, in_axis, 0, dim_vec.size()) {
        SbpSignatureBuilder()
            .Split("in", in_axis)
            .Split("out", in_axis < axis ? in_axis : in_axis + 1)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
