#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("bernoulli")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("seed", UserOpAttrType::kAtInt64, -1)
    .Attr<bool>("has_seed", UserOpAttrType::kAtBool, false)
    .Attr("dtype", UserOpAttrType::kAtDataType)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      *out_tensor->mut_shape() = in_tensor->shape();
      *out_tensor->mut_data_type() = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      for (int i = 0; i < in_tensor.shape().NumAxes(); ++i) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
