#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

Maybe<int64_t> FixSliceBegin(int64_t begin, int64_t dims) {
  begin = (begin >= 0) ? begin : begin + dims;
  CHECK_GE_OR_RETURN(begin, 0);
  CHECK_LT_OR_RETURN(begin, dims);
  return begin;
}

Maybe<int64_t> FixSliceEnd(int64_t end, int64_t dims) {
  end = end >= 0 ? end : end + dims;
  CHECK_GT_OR_RETURN(end, 0);
  return std::min(end, dims);
}

Maybe<void> InferOutputShape(user_op::InferContext* ctx) {
  Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  auto begin_vec = ctx->GetAttr<std::vector<int64_t>>("begin");
  auto end_vec = ctx->GetAttr<std::vector<int64_t>>("end");
  auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");
  // Don't support slice dim0 for now
  // so begin,end,stride must be 1 less than input's num of axes
  CHECK_EQ_OR_RETURN(in_shape->NumAxes(), begin_vec.size() + 1);
  CHECK_EQ_OR_RETURN(in_shape->NumAxes(), end_vec.size() + 1);
  CHECK_EQ_OR_RETURN(in_shape->NumAxes(), stride_vec.size() + 1);

  DimVector dim_vec(in_shape->NumAxes());
  FOR_RANGE(size_t, i, 0, dim_vec.size()) {
    CHECK_GT_OR_RETURN(in_shape->At(i), 0);
    if (i == 0) {
      dim_vec[i] = in_shape->At(i);
    } else {
      const int64_t begin = JUST(FixSliceBegin(begin_vec.at(i - 1), in_shape->At(i)));
      const int64_t end = JUST(FixSliceEnd(end_vec.at(i - 1), in_shape->At(i)));
      const int64_t stride = stride_vec.at(i - 1);
      CHECK_NE_OR_RETURN(begin, end);
      CHECK_NE_OR_RETURN(stride, 0);
      if (stride > 0) {
        CHECK_LT_OR_RETURN(begin, end);
      } else {
        CHECK_GT_OR_RETURN(begin, end);
      }
      int64_t align = (begin > end) ? 1 : -1;
      dim_vec[i] = (end - begin + align) / stride + 1;
    }
  }
  *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("slice_v2")
    .Input("x")
    .Output("y")
    .Attr("begin", UserOpAttrType::kAtListInt64)
    .Attr("end", UserOpAttrType::kAtListInt64)
    .Attr("stride", UserOpAttrType::kAtListInt64)
    .SetShapeInferFn(InferOutputShape)
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      ctx->BatchAxis4ArgNameAndIndex("y", 0)->set_value(0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder()
          .PartialSum(ctx->inputs())
          .PartialSum(ctx->outputs())
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("slice_grad_v2")
    .Input("dy")
    .Input("like")
    .Output("dx")
    .Attr("begin", UserOpAttrType::kAtListInt64)
    .Attr("end", UserOpAttrType::kAtListInt64)
    .Attr("stride", UserOpAttrType::kAtListInt64)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      auto begin_vec = ctx->GetAttr<std::vector<int64_t>>("begin");
      auto end_vec = ctx->GetAttr<std::vector<int64_t>>("end");
      auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");
      CHECK_EQ_OR_RETURN(like_shape->NumAxes(), begin_vec.size() + 1);
      CHECK_EQ_OR_RETURN(like_shape->NumAxes(), end_vec.size() + 1);
      CHECK_EQ_OR_RETURN(like_shape->NumAxes(), stride_vec.size() + 1);
      *ctx->Shape4ArgNameAndIndex("dx", 0) = *like_shape;
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      DataType* like_data_type = ctx->Dtype4ArgNameAndIndex("like", 0);
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("dy", 0), *like_data_type);
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *like_data_type;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      ctx->BatchAxis4ArgNameAndIndex("dx", 0)->set_value(0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder()
          .PartialSum(ctx->inputs())
          .PartialSum(ctx->outputs())
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder().PartialSum("dy").Broadcast("like").PartialSum("dx").Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder().Broadcast("dy").PartialSum("like").Broadcast("dx").Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("slice_v2")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("slice_grad_v2")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("like", op.input("x", 0))
                .Attr("begin", op.attr<std::vector<int64_t>>("begin"))
                .Attr("end", op.attr<std::vector<int64_t>>("end"))
                .Attr("stride", op.attr<std::vector<int64_t>>("stride"))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });
}  // namespace oneflow
