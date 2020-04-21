#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reshape_op_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

Maybe<Shape> GetLogicalOutTensorShape(const Shape& in_shape, const std::vector<int32_t>& shape) {
  size_t total_elem_dim_exclude_minus_1 = 1;
  bool has_minus_1 = false;
  bool minus_1_axis = -1;
  DimVector dim_vec;
  FOR_RANGE(int, axis, 0, shape.size()) {
    int64_t dim = shape.at(axis);
    dim_vec.push_back(dim);
    if (dim == -1) {
      OF_CHECK(has_minus_1 == false) << "only one `-1' supported";
      has_minus_1 = true;
      minus_1_axis = axis;
    } else if (dim > 0) {
      OF_CHECK_LE(dim, in_shape.elem_cnt()) << "invalid axis: " << axis << ", dim: " << dim;
      total_elem_dim_exclude_minus_1 *= dim;
      OF_CHECK_LE(total_elem_dim_exclude_minus_1, in_shape.elem_cnt())
          << "element number in reshape_conf is bigger than input blob";
    } else {
      OF_UNIMPLEMENTED() << "only positive number or -1 supported";
    }
  }
  OF_CHECK_EQ(in_shape.elem_cnt() % total_elem_dim_exclude_minus_1, 0);
  if (has_minus_1) {
    dim_vec[minus_1_axis] = in_shape.elem_cnt() / total_elem_dim_exclude_minus_1;
  } else {
    OF_CHECK_EQ(in_shape.elem_cnt(), total_elem_dim_exclude_minus_1)
        << "input blob's element number not equals reshape_conf";
  }
  return std::make_shared<Shape>(dim_vec);
}

REGISTER_USER_OP("reshape")
    .Input("in")
    .Output("out")
    .Attr("shape", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      const auto& shape = ctx->GetAttr<std::vector<int32_t>>("shape");
      CHECK_GE_OR_RETURN(shape.size(), 1);
      DimVector dim_vec = {shape.begin(), shape.end()};
      FOR_RANGE(int32_t, i, 0, dim_vec.size()) { CHECK_GE_OR_RETURN(dim_vec.at(i), 0); }
      const auto& sbp_parallel = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      const auto& parallel_ctx = ctx->parallel_ctx();

      if (sbp_parallel.has_split_parallel()) {
        const int64_t split_axis = sbp_parallel.split_parallel().axis();
        BalancedSplitter spliter(shape.at(split_axis), parallel_ctx.parallel_num());
        CHECK_GE_OR_RETURN(shape.at(split_axis), parallel_ctx.parallel_num());
        dim_vec.at(split_axis) = spliter.At(parallel_ctx.parallel_id()).size();
      }
      *out_shape = Shape(dim_vec);
      CHECK_EQ_OR_RETURN(out_shape->elem_cnt(), in_shape->elem_cnt());
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
      const auto& shape = ctx->GetAttr<std::vector<int32_t>>("shape");
      const auto& outshape = GetLogicalOutTensorShape(in_shape, shape)
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
    // const user_op::TensorDesc& tensor_desc = op.TensorDesc4ArgNameAndIndex("in", 0);
    // std::vector<int32_t> shape_vec = {tensor_desc.shape().dim_vec().begin(),
    //                                   tensor_desc.shape().dim_vec().end()};
    user_op::UserOpConfWrapper reshape_grad_op =
        builder.Op("reshape_like")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Input("like", op.input("in", 0))
            .Output("out")
            //.Attr<std::vector<int32_t>>("shape", shape_vec)
            .Build();
    op.BindGradTensorWithOpInput(reshape_grad_op.output("out", 0), "in", 0);
    AddOp(reshape_grad_op);
  }
});

}  // namespace oneflow