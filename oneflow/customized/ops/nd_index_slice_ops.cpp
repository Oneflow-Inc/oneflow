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

Maybe<void> InferScatterNdTensorDesc(user_op::InferContext* ctx) {
  Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
  Shape* updates_shape = ctx->Shape4ArgNameAndIndex("updates", 0);
  const Shape& params_shape = ctx->GetAttr<Shape>("shape");
  JUST(CheckScatterNdShape(params_shape, *indices_shape, *updates_shape));
  *ctx->Shape4ArgNameAndIndex("out", 0) = params_shape;
  *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("updates", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferScatterNdLikeTensorDesc(user_op::InferContext* ctx) {
  Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
  Shape* updates_shape = ctx->Shape4ArgNameAndIndex("updates", 0);
  Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
  JUST(CheckScatterNdShape(*like_shape, *indices_shape, *updates_shape));
  *ctx->Shape4ArgNameAndIndex("out", 0) = *like_shape;
  *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("updates", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorScatterNdOptTensorDesc(user_op::InferContext* ctx) {
  Shape* params_shape = ctx->Shape4ArgNameAndIndex("params", 0);
  Shape* updates_shape = ctx->Shape4ArgNameAndIndex("updates", 0);
  Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
  JUST(CheckScatterNdShape(*params_shape, *indices_shape, *updates_shape));
  *ctx->Shape4ArgNameAndIndex("out", 0) = *params_shape;
  *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("params", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetTensorScatterNdOptSbpSignatures(user_op::SbpContext* ctx) {
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
  SbpSignatureBuilder()
      .PartialSum("params", 0)
      .Broadcast("indices", 0)
      .PartialSum("updates", 0)
      .PartialSum("out", 0)
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorScatterNdOptBatchAxis(user_op::BatchAxisContext* ctx) {
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
      DimVector out_shape_vec(indices_shape->dim_vec().cbegin(),
                              indices_shape->dim_vec().cend() - 1);
      FOR_RANGE(int64_t, i, index_ndims, params_shape->NumAxes()) {
        out_shape_vec.push_back(params_shape->At(i));
      }
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape(out_shape_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("params", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      OptInt64* indices_batch_axis = ctx->BatchAxis4ArgNameAndIndex("indices", 0);
      if (indices_batch_axis->has_value()) {
        OF_CHECK_GE(indices_batch_axis->value(), 0);
        OF_CHECK_LT(
            indices_batch_axis->value(),
            ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes() - 1);
      }
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *indices_batch_axis;
      return Maybe<void>::Ok();
    })
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
      SbpSignatureBuilder()
          .PartialSum("params", 0)
          .Broadcast("indices", 0)
          .PartialSum("out", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("scatter_nd")
    .Input("indices")
    .Input("updates")
    .Output("out")
    .Attr("shape", UserOpAttrType::kAtShape)
    .SetTensorDescInferFn(InferScatterNdTensorDesc)
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      OF_CHECK(*ctx->BatchAxis4ArgNameAndIndex("indices", 0)
               == *ctx->BatchAxis4ArgNameAndIndex("updates", 0));
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& indices_desc =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
      int64_t indices_num_axes = indices_desc.shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
        SbpSignatureBuilder()
            .Split("indices", 0, i)
            .Split("updates", 0, i)
            .Broadcast("out", 0)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      const Shape& out_shape = ctx->GetAttr<Shape>("shape");
      int64_t index_ndims = indices_desc.shape().At(indices_num_axes - 1);
      int64_t slice_ndims = out_shape.NumAxes() - index_ndims;
      FOR_RANGE(int64_t, i, 0, slice_ndims) {
        SbpSignatureBuilder()
            .Broadcast("indices", 0)
            .Split("updates", 0, i + indices_num_axes - 1)
            .Split("out", 0, i + index_ndims)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      SbpSignatureBuilder()
          .PartialSum("updates", 0)
          .Broadcast("indices", 0)
          .PartialSum("out", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("scatter_nd_like")
    .Input("like")
    .Input("indices")
    .Input("updates")
    .Output("out")
    .SetTensorDescInferFn(InferScatterNdLikeTensorDesc)
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      OF_CHECK(*ctx->BatchAxis4ArgNameAndIndex("indices", 0)
               == *ctx->BatchAxis4ArgNameAndIndex("updates", 0));
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& indices_desc =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
      int64_t indices_num_axes = indices_desc.shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
        SbpSignatureBuilder()
            .Broadcast("like", 0)
            .Split("indices", 0, i)
            .Split("updates", 0, i)
            .Broadcast("out", 0)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      const Shape& out_shape = ctx->GetAttr<Shape>("shape");
      int64_t index_ndims = indices_desc.shape().At(indices_num_axes - 1);
      int64_t slice_ndims = out_shape.NumAxes() - index_ndims;
      FOR_RANGE(int64_t, i, 0, slice_ndims) {
        SbpSignatureBuilder()
            .Split("like", 0, i + index_ndims)
            .Broadcast("indices", 0)
            .Split("updates", 0, i + indices_num_axes - 1)
            .Split("out", 0, i + index_ndims)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      SbpSignatureBuilder()
          .PartialSum("like", 0)
          .PartialSum("updates", 0)
          .Broadcast("indices", 0)
          .PartialSum("out", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    });

REGISTER_USER_OP("tensor_scatter_nd_update")
    .Input("params")
    .Input("updates")
    .Input("indices")
    .Output("out")
    .SetTensorDescInferFn(InferTensorScatterNdOptTensorDesc)
    .SetBatchAxisInferFn(InferTensorScatterNdOptBatchAxis)
    .SetGetSbpFn(GetTensorScatterNdOptSbpSignatures);

REGISTER_USER_OP("tensor_scatter_nd_add")
    .Input("params")
    .Input("updates")
    .Input("indices")
    .Output("out")
    .SetTensorDescInferFn(InferTensorScatterNdOptTensorDesc)
    .SetBatchAxisInferFn(InferTensorScatterNdOptBatchAxis)
    .SetGetSbpFn(GetTensorScatterNdOptSbpSignatures);

REGISTER_USER_OP_GRAD("gather_nd")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("params", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scatter_nd_like")
                .Input("like", op.input("params", 0))
                .Input("updates", op.GetGradTensorWithOpOutput("out", 0))
                .Input("indices", op.input("indices", 0))
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

REGISTER_USER_OP_GRAD("tensor_scatter_nd_add")
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
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("out", 0), "params", 0);
      }
    });
}  // namespace oneflow
