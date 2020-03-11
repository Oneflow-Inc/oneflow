#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("arg_sort")
    .Input("in")
    .Output("out")
    .Attr("direction", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("in", 0);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // The current implementation can only do arg_sort in the last dimension and should use
      // Broadcast (by default) instead of Split for that dimension
      const int32_t num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
      if (num_axes > 1) {
        SbpSignatureBuilder()
            .Split(ctx->inputs(), 0)
            .Split(ctx->outputs(), 0)
            .MakeSplitSignatureListBuilder(num_axes - 1)
            .Build(ctx->sbp_sig_list());
      }
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      const std::string& direction = op_conf.attr<std::string>("direction");
      CHECK_OR_RETURN(direction == "ASCENDING" || direction == "DESCENDING");
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
