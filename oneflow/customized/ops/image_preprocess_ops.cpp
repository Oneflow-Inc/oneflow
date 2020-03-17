#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/image/image_util.h"

namespace oneflow {

REGISTER_USER_OP("Resize")
    .Input("in")
    .Output("out")
    .Attr<std::string>("color_space", UserOpAttrType::kAtString, "BGR")
    .Attr<std::string>("interp_type", UserOpAttrType::kAtString, "Linear")
    .Attr<std::string>("mag_filter", UserOpAttrType::kAtString, "Linear")
    .Attr<std::string>("min_filter", UserOpAttrType::kAtString, "Linear")
    .Attr<std::vector<float>>("max_size", UserOpAttrType::kAtListFloat, {0.0, 0.0})
    .Attr<float>("resize_longer", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("resize_shorter", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("resize_x", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("resize_y", UserOpAttrType::kAtFloat, 0.0)
    .Attr<int64_t>("seed", UserOpAttrType::kAtInt64, -1)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kTensorBuffer);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().At(0) >= 1);
      int64_t batch_size = in_tensor->shape().At(0);
      int64_t resize_x = ctx->GetAttr<float>("resize_x");
      int64_t resize_y = ctx->GetAttr<float>("resize_y");
      if (resize_x != 0 && resize_y != 0) {
        // resize_x -> W
        // resize_y -> H
        // shape = {H, W, c}
        std::string color_space = ctx->GetAttr<std::string>("color_space");
        int64_t c = ImageUtil::IsColor(color_space) ? 3 : 1;
        *out_tensor->mut_data_type() = DataType::kUInt8;
        *out_tensor->mut_shape() = Shape({batch_size, resize_y, resize_x, c});
      } else {
        *out_tensor->mut_data_type() = DataType::kTensorBuffer;
        *out_tensor->mut_shape() = Shape({batch_size});
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("in", 0, 0)
          .Split("out", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("in", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
