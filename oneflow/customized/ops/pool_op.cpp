#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

namespace {

typedef std::function<Maybe<void>(user_op::InferContext* ctx)> TensorDescInferFn;
typedef std::function<void(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp)>
    GenBackwardOpConfFn;

TensorDescInferFn MakeFwTensorDescInferFn(const int32_t dim) {
  return [dim](user_op::InferContext* ctx) -> Maybe<void> {
    const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
    const std::string data_format = ctx->GetAttr<std::string>("data_format");
    const std::string padding = ctx->GetAttr<std::string>("padding");
    const std::vector<int32_t> pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
    const std::vector<int32_t> strides = ctx->GetAttr<std::vector<int32_t>>("strides");

    CHECK_EQ_OR_RETURN(pool_size.size(), dim);
    for (int32_t pool_dim : pool_size) { CHECK_GT_OR_RETURN(pool_dim, 0); }
    CHECK_EQ_OR_RETURN(strides.size(), dim);
    for (int32_t stride_dim : strides) { CHECK_GT_OR_RETURN(stride_dim, 0); }

    const Params3D params_3d(dim, *x_shape, data_format, padding, pool_size, strides);
    user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
    *y_desc = *ctx->TensorDesc4ArgNameAndIndex("x", 0);
    *y_desc->mut_shape() = params_3d.GetYShape();
    return Maybe<void>::Ok();
  };
}

Maybe<void> BwTensorDescInferFn(user_op::InferContext* ctx) {
  *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *ctx->TensorDesc4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> FwBatchAxisInferFn(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> BwBatchAxisInferFn(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> BwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

GenBackwardOpConfFn MakeGenBackwardOpConfFn(const std::string& mode, const int32_t dim) {
  return [mode, dim](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
    if (op.NeedGenGradTensor4OpInput("x", 0)) {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
      user_op::UserOpConfWrapper grad_op =
          builder.Op(mode + "_pool_" + std::to_string(dim) + "d_grad")
              .Input("x", op.input("x", 0))
              .Input("y", op.output("y", 0))
              .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
              .Output("dx")
              .Attr("data_format", op.attr<std::string>("data_format"))
              .Attr("padding", op.attr<std::string>("padding"))
              .Attr("pool_size", op.attr<std::vector<int32_t>>("pool_size"))
              .Attr("strides", op.attr<std::vector<int32_t>>("strides"))
              .Build();
      op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
      AddOp(grad_op);
    }
  };
}

}  // namespace

REGISTER_USER_OP("avg_pool_1d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(MakeFwTensorDescInferFn(1))
    .SetBatchAxisInferFn(FwBatchAxisInferFn)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("avg_pool_1d_grad")
    .Input("x")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetBatchAxisInferFn(BwBatchAxisInferFn)
    .SetGetSbpFn(BwGetSbpFn);

REGISTER_USER_OP_GRAD("avg_pool_1d").SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("avg", 1));

REGISTER_USER_OP("avg_pool_2d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(MakeFwTensorDescInferFn(2))
    .SetBatchAxisInferFn(FwBatchAxisInferFn)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("avg_pool_2d_grad")
    .Input("x")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetBatchAxisInferFn(BwBatchAxisInferFn)
    .SetGetSbpFn(BwGetSbpFn);

REGISTER_USER_OP_GRAD("avg_pool_2d").SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("avg", 2));

REGISTER_USER_OP("avg_pool_3d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(MakeFwTensorDescInferFn(3))
    .SetBatchAxisInferFn(FwBatchAxisInferFn)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("avg_pool_3d_grad")
    .Input("x")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetBatchAxisInferFn(BwBatchAxisInferFn)
    .SetGetSbpFn(BwGetSbpFn);

REGISTER_USER_OP_GRAD("avg_pool_3d").SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("avg", 3));

REGISTER_USER_OP("max_pool_1d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(MakeFwTensorDescInferFn(1))
    .SetBatchAxisInferFn(FwBatchAxisInferFn)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("max_pool_1d_grad")
    .Input("x")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetBatchAxisInferFn(BwBatchAxisInferFn)
    .SetGetSbpFn(BwGetSbpFn);

REGISTER_USER_OP_GRAD("max_pool_1d").SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("max", 1));

REGISTER_USER_OP("max_pool_2d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(MakeFwTensorDescInferFn(2))
    .SetBatchAxisInferFn(FwBatchAxisInferFn)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("max_pool_2d_grad")
    .Input("x")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetBatchAxisInferFn(BwBatchAxisInferFn)
    .SetGetSbpFn(BwGetSbpFn);

REGISTER_USER_OP_GRAD("max_pool_2d").SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("max", 2));

REGISTER_USER_OP("max_pool_3d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(MakeFwTensorDescInferFn(3))
    .SetBatchAxisInferFn(FwBatchAxisInferFn)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("max_pool_3d_grad")
    .Input("x")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetBatchAxisInferFn(BwBatchAxisInferFn)
    .SetGetSbpFn(BwGetSbpFn);

REGISTER_USER_OP_GRAD("max_pool_3d").SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("max", 3));

}  // namespace oneflow
