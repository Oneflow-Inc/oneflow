#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_USER_OP("RawDecoder")
    .Input("in")
    .Output("out")
    .Attr("name", UserOpAttrType::kAtString)
    .Attr("shape", UserOpAttrType::kAtShape)
    .Attr("data_type", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_tensor->mut_shape() = Shape({5});
      *out_tensor->mut_data_type() = DataType::kFloat;
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("ImageDecoderRandomCrop")
    .Input("in")
    .Output("out")
    .Attr("name", UserOpAttrType::kAtString)
    .Attr<std::string>("color_space", UserOpAttrType::kAtString, "RGB")
    .Attr<int32_t>("num_attempts", UserOpAttrType::kAtInt32, 10)
    .Attr<int64_t>("seed", UserOpAttrType::kAtInt64, -1)
    .Attr<std::vector<float>>("random_area", UserOpAttrType::kAtListFloat, {0.08, 1.0})
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_tensor->mut_shape() = Shape({5});
      *out_tensor->mut_data_type() = DataType::kFloat;
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
