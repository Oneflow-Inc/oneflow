#include <cstdint>
#include <vector>
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/grad_registration.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/operator/reshape_op_util.h"

namespace oneflow {

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
    });
// .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
//     const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
//     //const auto& out_shape = ctx->
// });

REGISTER_USER_OP_GRAD("reshape").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    const user_op::TensorDesc& tensor_desc = op.TensorDesc4ArgNameAndIndex("in", 0);
    std::vector<int32_t> shape_vec = {tensor_desc.shape().dim_vec().begin(),
                                      tensor_desc.shape().dim_vec().end()};
    user_op::UserOpConfWrapper reshape_grad_op =
        builder.Op("reshape")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Output("out")
            .Attr<std::vector<int32_t>>("shape", shape_vec)
            .Build();
    op.BindGradTensorWithOpInput(reshape_grad_op.output("out", 0), "in", 0);
    AddOp(reshape_grad_op);
  }
});

}  // namespace oneflow