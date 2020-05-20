#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_USER_OP("ofrecord_raw_decoder")
    .Input("in")
    .Output("out")
    .Attr("name", UserOpAttrType::kAtString)
    .Attr("shape", UserOpAttrType::kAtShape)
    .Attr("data_type", UserOpAttrType::kAtInt64)
    .Attr<bool>("dim1_varying_length", UserOpAttrType::kAtBool, false)
    .Attr<bool>("auto_zero_padding", UserOpAttrType::kAtBool, false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kOFRecord);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().At(0) >= 1);
      Shape conf_shape = ctx->GetAttr<Shape>("shape");
      DimVector dim_vec(1 + conf_shape.NumAxes());
      dim_vec[0] = in_tensor->shape().At(0);
      for (int i = 1; i < dim_vec.size(); ++i) { dim_vec[i] = conf_shape.At(i - 1); }
      *out_tensor->mut_shape() = Shape(dim_vec);
      int64_t data_type = ctx->GetAttr<int64_t>("data_type");
      *out_tensor->mut_data_type() = static_cast<DataType>(data_type);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("in", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("ofrecord_image_decoder_random_crop")
    .Input("in")
    .Output("out")
    .Attr("name", UserOpAttrType::kAtString)
    .Attr<std::string>("color_space", UserOpAttrType::kAtString, "BGR")
    .Attr<int32_t>("num_attempts", UserOpAttrType::kAtInt32, 10)
    .Attr<int64_t>("seed", UserOpAttrType::kAtInt64, -1)
    .Attr<bool>("has_seed", UserOpAttrType::kAtBool, false)
    .Attr<std::vector<float>>("random_area", UserOpAttrType::kAtListFloat, {0.08, 1.0})
    .Attr<std::vector<float>>("random_aspect_ratio", UserOpAttrType::kAtListFloat, {0.75, 1.333333})
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kOFRecord);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().At(0) >= 1);
      *out_tensor->mut_shape() = in_tensor->shape();
      *out_tensor->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("in", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
