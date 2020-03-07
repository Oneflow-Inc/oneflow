#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferScatterNdOptShape(user_op::InferContext* ctx) {
  Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
  Shape* updates_shape = ctx->Shape4ArgNameAndIndex("updates", 0);
  Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
  int64_t segm_dim = indices_shape->At(indices_shape->NumAxes() - 1);
  OF_CHECK_LE(segm_dim, in_shape->NumAxes());
  FOR_RANGE(int64_t, i, 0, updates_shape->NumAxes() - 1) {
    OF_CHECK_EQ(updates_shape->At(i), indices_shape->At(i));
  }
  OF_CHECK_EQ(updates_shape->NumAxes() - indices_shape->NumAxes() + 1,
              in_shape->NumAxes() - segm_dim);
  for (int64_t i = updates_shape->NumAxes() - 1, j = segm_dim; j < in_shape->NumAxes(); ++i, ++j) {
    OF_CHECK_EQ(updates_shape->At(i), in_shape->At(j));
  }
  *ctx->Shape4ArgNameAndIndex("out", 0) = *in_shape;
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("gather_nd")
    .Input("x")
    .Input("indices")
    .Output("y")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      int64_t segm_dim = indices_shape->At(indices_shape->NumAxes() - 1);
      OF_CHECK_LE(segm_dim, x_shape->NumAxes());
      DimVector y_shape_vec = indices_shape->dim_vec();
      y_shape_vec.pop_back();
      FOR_RANGE(int64_t, i, segm_dim, x_shape->NumAxes()) { y_shape_vec.push_back(x_shape->At(i)); }
      *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(y_shape_vec);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("scatter_nd_update")
    .Input("in")
    .Input("updates")
    .Input("indices")
    .Output("out")
    .SetShapeInferFn(InferScatterNdOptShape)
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("scatter_nd_add")
    .Input("in")
    .Input("updates")
    .Input("indices")
    .Output("out")
    .SetShapeInferFn(InferScatterNdOptShape)
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("gather_nd")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder zero_grad_builder(op.op_name() + "_zero_grad");
        user_op::UserOpConfWrapper zero_grad_op =
            zero_grad_builder.Op("zero_like").Input("like", op.input("x", 0)).Output("out").Build();
        AddOp(zero_grad_op);
        user_op::UserOpConfWrapperBuilder grad_builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            grad_builder.Op("scatter_nd_add")
                .Input("in", zero_grad_op.output("out", 0))
                .Input("updates", op.GetGradTensorWithOpOutput("y", 0))
                .Input("indices", op.input("indices", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "x", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("scatter_nd_update")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("updates", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_updates_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("gather_nd")
                                                 .Input("x", op.GetGradTensorWithOpOutput("out", 0))
                                                 .Input("indices", op.input("indices", 0))
                                                 .Output("y")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "updates", 0);
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder zero_grad_builder(op.op_name() + "_zero_updates");
        user_op::UserOpConfWrapper zero_grad_op = zero_grad_builder.Op("zero_like")
                                                      .Input("like", op.input("updates", 0))
                                                      .Output("out")
                                                      .Build();
        AddOp(zero_grad_op);
        user_op::UserOpConfWrapperBuilder grad_builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            grad_builder.Op("scatter_nd_update")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Input("updates", zero_grad_op.output("out", 0))
                .Input("indices", op.input("indices", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
