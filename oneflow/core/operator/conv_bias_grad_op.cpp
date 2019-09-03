#include "oneflow/core/operator/conv_bias_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

const PbMessage& ConvBiasGradOp::GetCustomizedConf() const {
  return op_conf().conv_bias_grad_conf();
}

void ConvBiasGradOp::InitFromOpConf() {
  CHECK(op_conf().has_conv_bias_grad_conf());
  EnrollInputBn("dy", false);
  EnrollOutputBn("bias_diff", false);
}

Maybe<void> ConvBiasGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ConvBiasGradOpConf& conf = this->op_conf().conv_bias_grad_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  BlobDesc* bias_diff = GetBlobDesc4BnInOp("bias_diff");
  CHECK_GE_OR_RETURN(conf.num_spatial_dims(), 1);
  CHECK_LE_OR_RETURN(conf.num_spatial_dims(), 3);
  CHECK_EQ_OR_RETURN(dy->shape().NumAxes(), conf.num_spatial_dims() + 2);
  bias_diff->set_data_type(dy->data_type());
  if (conf.data_format() == "channels_first") {
    bias_diff->mut_shape() = Shape({dy->shape().At(1)});
  } else if (conf.data_format() == "channels_last") {
    bias_diff->mut_shape() = Shape({dy->shape().At(dy->shape().NumAxes() - 1)});
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

Maybe<void> ConvBiasGradOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(BatchAxis4BnInOp("dy")->has_value());
  BatchAxis4BnInOp("bias_diff")->clear_value();
  return Maybe<void>::Ok();
}

void ConvBiasGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("dy", 0)
      .PartialSum("bias_diff")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kConvBiasGradConf, ConvBiasGradOp);

}  // namespace oneflow
