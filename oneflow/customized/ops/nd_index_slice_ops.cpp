#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> CheckScatterNdShape(const Shape& params_shape, const Shape& indices_shape,
                                const Shape& updates_shape) {
  int64_t batch_ndims = indices_shape.NumAxes() - 1;
  int64_t index_ndims = indices_shape.At(batch_ndims);
  OF_CHECK_LE(batch_ndims, updates_shape.NumAxes());
  OF_CHECK_LE(index_ndims, params_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, batch_ndims) { OF_CHECK_EQ(updates_shape.At(i), indices_shape.At(i)); }
  int64_t slice_ndims = params_shape.NumAxes() - index_ndims;
  OF_CHECK_EQ(slice_ndims, updates_shape.NumAxes() - batch_ndims);
  FOR_RANGE(int64_t, i, 0, slice_ndims) {
    OF_CHECK_EQ(updates_shape.At(i + batch_ndims), params_shape.At(i + index_ndims));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferScatterNdOptTensorDesc(user_op::InferContext* ctx) {
  Shape* params_shape = ctx->Shape4ArgNameAndIndex("params", 0);
  Shape* updates_shape = ctx->Shape4ArgNameAndIndex("updates", 0);
  Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
  JUST(CheckScatterNdShape(*params_shape, *indices_shape, *updates_shape));
  *ctx->Shape4ArgNameAndIndex("out", 0) = *params_shape;
  *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("params", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetScatterNdOptSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& params_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("params", 0);
  const user_op::TensorDesc& indices_desc =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
  int64_t indices_num_axes = indices_desc.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
    SbpSignatureBuilder()
        .Broadcast("params", 0)
        .Split("indices", 0, i)
        .Split("updates", 0, i)
        .Broadcast("out", 0)
        .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  }
  int64_t index_ndims = indices_desc.shape().At(indices_num_axes - 1);
  FOR_RANGE(int64_t, i, index_ndims, params_desc.shape().NumAxes()) {
    SbpSignatureBuilder()
        .Split("params", 0, i)
        .Broadcast("indices", 0)
        .Split("updates", 0, i - index_ndims + indices_num_axes - 1)
        .Split("out", 0, i)
        .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferGatherScatterNdBatchAxis(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("indices", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferScatterNdOptBatchAxis(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("params", 0);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("gather_nd")
    .Input("params")
    .Input("indices")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* params_shape = ctx->Shape4ArgNameAndIndex("params", 0);
      Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      int64_t index_ndims = indices_shape->At(indices_shape->NumAxes() - 1);
      OF_CHECK_LE(index_ndims, params_shape->NumAxes());
      DimVector out_shape_vec = indices_shape->dim_vec();
      out_shape_vec.pop_back();
      FOR_RANGE(int64_t, i, index_ndims, params_shape->NumAxes()) {
        out_shape_vec.push_back(params_shape->At(i));
      }
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape(out_shape_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("params", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(InferGatherScatterNdBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& params_desc =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("params", 0);
      const user_op::TensorDesc& indices_desc =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
      int64_t indices_num_axes = indices_desc.shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
        SbpSignatureBuilder()
            .Broadcast("params", 0)
            .Split("indices", 0, i)
            .Split("out", 0, i)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      int64_t index_ndims = indices_desc.shape().At(indices_num_axes - 1);
      FOR_RANGE(int64_t, i, index_ndims, params_desc.shape().NumAxes()) {
        SbpSignatureBuilder()
            .Split("params", 0, i)
            .Broadcast("indices", 0)
            .Split("out", 0, i - index_ndims + indices_num_axes - 1)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("scatter_nd")
    .Input("indices")
    .Input("updates")
    .Output("out")
    .Attr("shape", UserOpAttrType::kAtShape)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      Shape* updates_shape = ctx->Shape4ArgNameAndIndex("updates", 0);
      Shape params_shape = ctx->GetAttr<Shape>("shape");
      JUST(CheckScatterNdShape(params_shape, *indices_shape, *updates_shape));
      *ctx->Shape4ArgNameAndIndex("out", 0) = params_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("updates", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(InferGatherScatterNdBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { TODO(); });

REGISTER_USER_OP("tensor_scatter_nd_update")
    .Input("params")
    .Input("updates")
    .Input("indices")
    .Output("out")
    .SetTensorDescInferFn(InferScatterNdOptTensorDesc)
    .SetBatchAxisInferFn(InferScatterNdOptBatchAxis)
    .SetGetSbpFn(GetScatterNdOptSbpSignatures);

REGISTER_USER_OP("scatter_nd_add")
    .Input("params")
    .Input("updates")
    .Input("indices")
    .Output("out")
    .SetTensorDescInferFn(InferScatterNdOptTensorDesc)
    .SetBatchAxisInferFn(InferScatterNdOptBatchAxis)
    .SetGetSbpFn(GetScatterNdOptSbpSignatures);

REGISTER_USER_OP_GRAD("gather_nd")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("params", 0)) {
        const user_op::TensorDesc& params_desc = op.TensorDesc4ArgNameAndIndex("params", 0);
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scatter_nd")
                .Input("updates", op.GetGradTensorWithOpOutput("out", 0))
                .Input("indices", op.input("indices", 0))
                .Attr("shape", params_desc.shape())
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "params", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("scatter_nd")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("updates", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("gather_nd")
                .Input("params", op.GetGradTensorWithOpOutput("out", 0))
                .Input("indices", op.input("indices", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "updates", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("tensor_scatter_nd_update")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("updates", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_updates_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("gather_nd")
                .Input("params", op.GetGradTensorWithOpOutput("out", 0))
                .Input("indices", op.input("indices", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "updates", 0);
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("params", 0)) {
        user_op::UserOpConfWrapperBuilder zero_grad_builder(op.op_name() + "_zero_updates");
        user_op::UserOpConfWrapper zero_grad_op = zero_grad_builder.Op("zero_like")
                                                      .Input("like", op.input("updates", 0))
                                                      .Output("out")
                                                      .Build();
        AddOp(zero_grad_op);
        user_op::UserOpConfWrapperBuilder grad_builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            grad_builder.Op("tensor_scatter_nd_update")
                .Input("params", op.GetGradTensorWithOpOutput("out", 0))
                .Input("updates", zero_grad_op.output("out", 0))
                .Input("indices", op.input("indices", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "params", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
