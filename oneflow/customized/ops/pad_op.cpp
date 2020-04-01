#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_USER_OP("pad")
    .Input("x")
    .Output("y")
    .Attr("paddings", UserOpAttrType::kAtListInt64)
    .Attr("constant_value", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      auto paddings_vec = ctx->GetAttr<std::vector<int64_t>>("paddings");
      CHECK_EQ(paddings_vec.size(), 2 * x_shape->NumAxes());
      DimVector y_dim_vec(x_shape->NumAxes());
      FOR_RANGE(int64_t, i, 0, x_shape->NumAxes()){
        y_dim_vec[i] = x_shape->At(i) + paddings_vec[2 * i] + paddings_vec[2 * i + 1];
      }
      *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(y_dim_vec);
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO: check sbp
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("pad_grad")
    .Input("dy")
    .Output("dx")
    .Attr("paddings", UserOpAttrType::kAtListInt64)
    .Attr("constant_value", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      auto paddings_vec = ctx->GetAttr<std::vector<int64_t>>("paddings");
      CHECK_EQ(paddings_vec.size(), 2 * dy_shape->NumAxes());
      DimVector dx_dim_vec(dy_shape->NumAxes());
      FOR_RANGE(int64_t, i, 0, dy_shape->NumAxes()){
        dx_dim_vec[i] = dy_shape->At(i) - paddings_vec[2 * i] - paddings_vec[2 * i + 1];
      }
      *ctx->Shape4ArgNameAndIndex("dx", 0) = Shape(dx_dim_vec);
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO: check sbp
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("dy", 0, 0)
          .Split("dx", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("pad")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("pad_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr("constant_value", op.attr<float>("constant_value"))
                .Attr("paddings", op.attr<std::vector<int64_t>>("paddings"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("pad_grad")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("pad")
                .Input("x", op.GetGradTensorWithOpOutput("dy", 0))
                .Output("y")
                .Attr("constant_value", op.attr<float>("constant_value"))
                .Attr("paddings", op.attr<std::vector<int64_t>>("paddings"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "dx", 0);
        AddOp(grad_op);
      }
    });


}  // namespace oneflow
