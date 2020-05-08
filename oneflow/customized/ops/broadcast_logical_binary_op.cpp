#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("binary")
    .Input("x")
    .Input("y")
    .Output("z")
    .Attr("binary_math_type", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* z_shape = ctx->Shape4ArgNameAndIndex("z", 0);
      CHECK(*y_shape == *x_shape);
      *z_shape = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor_x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      // const user_op::TensorDesc& tensor_y = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor_x.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("broadcast_binary")
    .Input("x")
    .Input("y")
    .Output("z")
    .Attr("binary_math_type", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* z_shape = ctx->Shape4ArgNameAndIndex("z", 0);
      CHECK(*y_shape == *x_shape);
      *z_shape = *x_shape;
      DataType* z_dtype = ctx->Dtype4ArgNameAndIndex("z", 0);
      *z_dtype = kInt8;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor_x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      const user_op::TensorDesc& tensor_y = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor_x.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
