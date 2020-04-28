#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reshape_op_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

REGISTER_USER_OP("reshape")
    .Input("in")
    .Output("out")
    .Attr("shape", UserOpAttrType::kAtShape)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      const Shape& shape = ctx->GetAttr<Shape>("shape");
      CHECK_GE_OR_RETURN(shape.NumAxes(), 1);
      DimVector dim_vec = {shape.dim_vec().begin(), shape.dim_vec().end()};
      FOR_RANGE(int32_t, i, 0, dim_vec.size()) { CHECK_GE_OR_RETURN(dim_vec.at(i), 0); }
      const auto& sbp_parallel = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      const auto& parallel_ctx = ctx->parallel_ctx();
      if (sbp_parallel.has_split_parallel()) {
        const int64_t split_axis = sbp_parallel.split_parallel().axis();
        BalancedSplitter spliter(shape.dim_vec().at(split_axis), parallel_ctx.parallel_num());
        CHECK_GE_OR_RETURN(shape.dim_vec().at(split_axis), parallel_ctx.parallel_num());
        dim_vec.at(split_axis) = spliter.At(parallel_ctx.parallel_id()).size();
      }
      *out_shape = Shape(dim_vec);
      CHECK_EQ_OR_RETURN(out_shape->elem_cnt(), in_shape->elem_cnt());
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
      const Shape& shape = ctx->GetAttr<Shape>("shape");
      ShapeProto shape_proto;
      shape.ToProto(&shape_proto);
      const auto& outshape = ReshapeOpUtil::GetLogicalOutBlobShape(in_shape, shape_proto)
                                 .Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
      return ReshapeOpUtil::GetReshapeSbpSignatures(
          in_shape, *outshape, StdVec2PbRpf<std::string>({GenRepeatedBn("in", 0)}),
          StdVec2PbRpf<std::string>({GenRepeatedBn("out", 0)}), ctx->parallel_num(),
          ctx->sbp_sig_list());
    });

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

}  // namespace oneflow
