#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"
namespace oneflow {

REGISTER_USER_OP("pool")
    .Input("x")
    .Output("y")
    .Attr("dim", UserOpAttrType::kAtInt32)
    .Attr("pooling_type", UserOpAttrType::kAtString)
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const int32_t dim = ctx->GetAttr<int32_t>("dim");
      const std::string data_format = ctx->GetAttr<std::string>("data_format");
      const std::string padding = ctx->GetAttr<std::string>("padding");
      const std::vector<int32_t> pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
      const std::vector<int32_t> strides = ctx->GetAttr<std::vector<int32_t>>("strides");
      const Params3D params_3d(dim, *x_shape, data_format, padding, pool_size, strides);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      *y_shape = params_3d.GetYShape();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("y", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("pool_grad")
    .Input("x")
    .Input("dy")
    .Output("dx")
    .Attr("alpha", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK(*dy_shape == *x_shape);
      *dx_shape = *dy_shape;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("dy", 0, 0)
          .Split("dx", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      SbpSignatureBuilder().Broadcast("x", 0).PartialSum("dy", 0).PartialSum("dx", 0).Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("pool").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("pool_grad")
                                             .Input("x", op.input("x", 0))
                                             .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                             .Output("dx")
                                             .Attr("alpha", op.attr<float>("alpha"))
                                             .Build();
    op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
    AddOp(grad_op);
  }
});

}  // namespace oneflow
