#include "oneflow/core/operator/normalization_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void NormalizationOp::InitFromOpConf() {
  const NormalizationOpConf& conf = op_conf().normalization_conf();
#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) { CHECK_GE(conf.epsilon(), CUDNN_BN_MIN_EPSILON); }
#endif
  CHECK_GE(conf.momentum(), 0);
  CHECK_LE(conf.momentum(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollInputBn("moving_mean")->set_is_mutable(conf.is_training());
  EnrollInputBn("moving_variance")->set_is_mutable(conf.is_training());
  if (conf.scale()) {
    CHECK(conf.has_gamma());
    EnrollInputBn("gamma");
  } else if (DevIsGpuAndEnableCudnn()) {
    EnrollConstBufBn("gamma");
  } else {
    UNIMPLEMENTED();
  }
  if (conf.center()) {
    CHECK(conf.has_beta());
    EnrollInputBn("beta");
  } else if (DevIsGpuAndEnableCudnn()) {
    EnrollConstBufBn("beta");
  } else {
    UNIMPLEMENTED();
  }
  if (conf.is_training()) {
    EnrollOutputBn("mean", false);
    EnrollOutputBn("inv_variance", false);
  }
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

Maybe<void> NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NormalizationOpConf& conf = op_conf().normalization_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const DataType data_type = in->data_type();
  *GetBlobDesc4BnInOp("out") = *in;
  int32_t axis = conf.axis();
  OF_CHECK_GE(axis, 0);
  OF_CHECK_LT(axis, in->shape().NumAxes());
  const Shape param_shape({in->shape().At(axis)});
  const std::function<void(const std::string&)> CheckParamBlobDesc =
      [&](const std::string& bn) -> Maybe<void> {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);
    if (blob_desc != nullptr) {
      CHECK_EQ_OR_RETURN(blob_desc->data_type(), data_type);
      CHECK_EQ_OR_RETURN(blob_desc->shape(), param_shape);
    }
    return Maybe<void>::Ok();
  };
  const std::function<void(const std::string&)> SetParamBlobDesc =
      [&](const std::string& bn) -> Maybe<void> {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);
    if (blob_desc != nullptr) {
      blob_desc->set_data_type(data_type);
      blob_desc->mut_shape() = param_shape;
    }
    return Maybe<void>::Ok();
  };
  CheckParamBlobDesc("moving_mean");
  CheckParamBlobDesc("moving_variance");
  if (conf.center()) {
    CheckParamBlobDesc("beta");
  } else {
    SetParamBlobDesc("beta");
  }
  if (conf.scale()) {
    CheckParamBlobDesc("gamma");
  } else {
    SetParamBlobDesc("gamma");
  }
  if (conf.is_training()) {
    SetParamBlobDesc("mean");
    SetParamBlobDesc("inv_variance");
  }
  return Maybe<void>::Ok();
}

Maybe<void> NormalizationOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
  if (op_conf().normalization_conf().is_training()) {
    BatchAxis4BnInOp("mean")->clear_value();
    BatchAxis4BnInOp("inv_variance")->clear_value();
  }
  return Maybe<void>::Ok();
}

Maybe<void> NormalizationOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Broadcast(output_bns())
      .Split("in", 0)
      .Split("out", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
