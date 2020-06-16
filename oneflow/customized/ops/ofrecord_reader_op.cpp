#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("OFRecordReader")
    .Output("out")
    .Attr("data_dir", UserOpAttrType::kAtString)
    .Attr("data_part_num", UserOpAttrType::kAtInt32)
    .Attr("batch_size", UserOpAttrType::kAtInt32)
    .Attr<std::string>("part_name_prefix", UserOpAttrType::kAtString, "part-")
    .Attr<int32_t>("part_name_suffix_length", UserOpAttrType::kAtInt32, -1)
    .Attr<bool>("random_shuffle", UserOpAttrType::kAtBool, false)
    .Attr<int64_t>("seed", UserOpAttrType::kAtInt64, -1)
    .Attr<int32_t>("shuffle_buffer_size", UserOpAttrType::kAtInt32, 1024)
    .Attr<bool>("shuffle_after_epoch", UserOpAttrType::kAtBool, false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      int32_t local_batch_size = ctx->Attr<int32_t>("batch_size");
      const SbpParallel& sbp = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      int64_t parallel_num = ctx->parallel_ctx().parallel_num();
      if (sbp.has_split_parallel() && parallel_num > 1) {
        CHECK_EQ_OR_RETURN(local_batch_size % parallel_num, 0);
        local_batch_size /= parallel_num;
      }
      *out_tensor->mut_shape() = Shape({local_batch_size});
      *out_tensor->mut_data_type() = DataType::kOFRecord;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
