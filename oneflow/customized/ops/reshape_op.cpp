#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reshape_op_util.h"

namespace oneflow {

namespace {

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  const Shape& shape = ctx->GetAttr<Shape>("shape");
  ShapeProto shape_proto;
  shape.ToProto(&shape_proto);
  const auto& outshape = JUST(ReshapeOpUtil::GetLogicalOutBlobShape(in_shape, shape_proto));
  return ReshapeOpUtil::GetReshapeSbpSignatures(
      in_shape, *outshape, StdVec2PbRpf<std::string>({GenRepeatedBn("in", 0)}),
      StdVec2PbRpf<std::string>({GenRepeatedBn("out", 0)}), ctx->parallel_num(),
      ctx->sbp_sig_list());
}

Maybe<void> TensorDescInferFn(user_op::InferContext* ctx) {
  const Shape& shape = ctx->GetAttr<Shape>("shape");
  const user_op::TensorDesc* in_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  user_op::TensorDesc* out_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  const Shape& in_shape = in_tensor_desc->shape();
  Shape* out_shape = out_tensor_desc->mut_shape();
  CHECK_OR_RETURN(in_tensor_desc->is_dynamic() == false);
  *out_tensor_desc = *in_tensor_desc;
  CHECK_GE_OR_RETURN(shape.NumAxes(), 1);
  DimVector dim_vec = {shape.dim_vec().begin(), shape.dim_vec().end()};
  FOR_RANGE(int32_t, i, 0, dim_vec.size()) { CHECK_GT_OR_RETURN(dim_vec.at(i), 0); }
  const auto& sbp_parallel = ctx->SbpParallel4ArgNameAndIndex("out", 0);
  const auto& parallel_ctx = ctx->parallel_ctx();
  if (sbp_parallel.has_split_parallel()) {
    const int64_t split_axis = sbp_parallel.split_parallel().axis();
    BalancedSplitter spliter(shape.dim_vec().at(split_axis), parallel_ctx.parallel_num());
    CHECK_GE_OR_RETURN(shape.dim_vec().at(split_axis), parallel_ctx.parallel_num());
    dim_vec.at(split_axis) = spliter.At(parallel_ctx.parallel_id()).size();
  }
  *out_shape = Shape(dim_vec);
  CHECK_EQ_OR_RETURN(out_shape->elem_cnt(), in_shape.elem_cnt());
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("reshape")
    .Input("in")
    .Output("out")
    .Attr("shape", UserOpAttrType::kAtShape)
    .SetTensorDescInferFn(TensorDescInferFn)
    .SetGetSbpFn(GetSbpFn);

REGISTER_USER_OP_GRAD("reshape").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper reshape_grad_op =
        builder.Op("reshape_like")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Input("like", op.input("in", 0))
            .Output("out")
            .Build();
    op.BindGradTensorWithOpInput(reshape_grad_op.output("out", 0), "in", 0);
    AddOp(reshape_grad_op);
  }
});

}  // namespace
}  // namespace oneflow
