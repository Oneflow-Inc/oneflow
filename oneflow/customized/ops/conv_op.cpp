#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/ops/nn_util.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4Conv(user_op::InferContext* ctx) {
  const int32_t NDims = 2;
  const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  CHECK_EQ(NDims + 2, in->shape().NumAxes());

  auto data_format = ctx->GetAttr<std::string>("data_format");
  auto kernel_size = ctx->GetAttr<std::vector<int32_t>>("kernel_size");
  int32_t filters = ctx->GetAttr<int32_t>("filters");
  size_t idx_offset = IdxOffset(data_format);

  {
    auto padding = ctx->GetAttr<std::string>("padding");
    auto dilation_rate = ctx->GetAttr<std::vector<int32_t>>("dilation_rate");
    auto strides = ctx->GetAttr<std::vector<int32_t>>("strides");

    user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
    DimVector out_shape(in->shape().dim_vec());
    for (int32_t i = 0; i < NDims; ++i) {
      CalcOutAndPadding(in->shape().At(idx_offset + i), kernel_size.at(i), dilation_rate.at(i),
                        strides.at(i), padding, &out_shape.at(idx_offset + i), nullptr, nullptr);
    }
    *out = *in;
    *out->mut_shape() = Shape(out_shape);
  }

  {
    int32_t groups = ctx->GetAttr<int32_t>("groups");
    CHECK_GT_OR_RETURN(groups, 0);
    CHECK_LE_OR_RETURN(groups, filters);
    CHECK_EQ_OR_RETURN(filters % groups, 0);

    DimVector weight_shape(in->shape().dim_vec());
    weight_shape.at(0) = filters;
    if (data_format == "channels_first") {
      CHECK_LE_OR_RETURN(groups, weight_shape.at(1));
      CHECK_EQ_OR_RETURN(weight_shape.at(1) % groups, 0);
      weight_shape.at(1) = weight_shape.at(1) / groups;
    } else if (data_format == "channels_last") {
      CHECK_LE_OR_RETURN(groups, weight_shape.at(NDims + 1));
      CHECK_EQ_OR_RETURN(weight_shape.at(NDims + 1) % groups, 0);
      weight_shape.at(NDims + 1) = weight_shape.at(NDims + 1) / groups;
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    for (size_t i = 0; i < NDims; ++i) { weight_shape.at(idx_offset + i) = kernel_size.at(i); }

    const user_op::TensorDesc* weight = ctx->TensorDesc4ArgNameAndIndex("weight", 0);
    CHECK_EQ(weight->shape(), Shape(weight_shape));
  }

  const user_op::TensorDesc* bias = ctx->TensorDesc4ArgNameAndIndex("bias", 0);
  if (bias != nullptr) { CHECK_EQ_OR_RETURN(bias->shape(), Shape({filters})); }
  return Maybe<void>::Ok();
}

Maybe<void> InferBatchAxis4Conv(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures4Conv(user_op::SbpContext* ctx) {
  // TODO(niuchong) : handle bias_multiplier
  bool has_bias = false;
  for (const auto& pair : ctx->inputs()) {
    if (pair.first == "bias") {
      CHECK_EQ_OR_RETURN(0, pair.second);
      has_bias = true;
      break;
    }
  }

  if (has_bias) {
    SbpSignatureBuilder()
        .Split("in", 0, 0)
        .Broadcast("weight", 0)
        .Broadcast("bias", 0)
        .Split("out", 0, 0)
        .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  } else {
    SbpSignatureBuilder()
        .Split("in", 0, 0)
        .Broadcast("weight", 0)
        .Split("out", 0, 0)
        .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("conv2d")
    .Input("in")
    .Input("weight")
    .OptionalInput("bias")
    .OptionalInput("bias_multiplier")
    .Output("out")
    .Attr("filters", UserOpAttrType::kAtInt32)
    .Attr<std::string>("padding", UserOpAttrType::kAtString, "valid")
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("kernel_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .Attr("dilation_rate", UserOpAttrType::kAtListInt32)
    .Attr<int32_t>("groups", UserOpAttrType::kAtListInt32, 1)
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& def,
                       const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
      std::string data_format = conf.attr<std::string>("data_format");
      if (data_format == "channels_first" || data_format == "channels_last") {
        return Maybe<void>::Ok();
      }
      return oneflow::Error::CheckFailed()
             << "data_format value: " << data_format << " for Conv op is illegal";
    })
    .SetTensorDescInferFn(InferTensorDesc4Conv)
    .SetBatchAxisInferFn(InferBatchAxis4Conv)
    .SetGetSbpFn(GetSbpSignatures4Conv);

}  // namespace oneflow
