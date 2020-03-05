#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("argmax")
    .Input("in")
    .Output("out")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      auto dim_vec = ctx->Shape4ArgNameAndIndex("in", 0)->dim_vec();
      dim_vec.pop_back();
      *ctx->Shape4ArgNameAndIndex("out", 0) =
          dim_vec.empty() ? Shape({1}) : Shape(std::move(dim_vec));
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(in_desc.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
