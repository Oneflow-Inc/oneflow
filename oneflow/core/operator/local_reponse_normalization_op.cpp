#include "oneflow/core/operator/local_reponse_normalization_op.h"

namespace oneflow {

void LocalResponseNormalizationOp::InitFromOpConf() {
  CHECK(op_conf().has_local_response_normalization_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollTmpBn("padded_square");
  EnrollTmpBn("normalize_coef");
}

const PbMessage& LocalResponseNormalizationOp::GetCustomizedConf() const {
  return op_conf().local_response_normalization_conf();
}

Maybe<void> LocalResponseNormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const LocalResponseNormalizationOpConf conf = op_conf().local_response_normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ_OR_RETURN(in_blob_desc->data_type(), GlobalJobDesc().DefaultDataType());
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;

  if (device_type() == DeviceType::kCPU) {
    if (conf.data_format() == "channels_last") {
      GetBlobDesc4BnInOp("padded_square")->mut_shape() =
          Shape({in_blob_desc->shape().At(3) + 2 * conf.depth_radius()});
    } else if (conf.data_format() == "channels_first") {
      GetBlobDesc4BnInOp("padded_square")->mut_shape() =
          Shape({1, in_blob_desc->shape().At(1) + 2 * conf.depth_radius(),
                 in_blob_desc->shape().At(2), in_blob_desc->shape().At(3)});
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    GetBlobDesc4BnInOp("normalize_coef")->mut_shape() = in_blob_desc->shape();
  } else if (device_type() == DeviceType::kGPU) {
    CHECK_STREQ_OR_RETURN(conf.data_format().c_str(), "channels_first");
    // cudnn requirements
    CHECK_GE_OR_RETURN(conf.bias(), 1e-5);
    CHECK_GE_OR_RETURN(conf.beta(), 0.01);
    CHECK_GE_OR_RETURN(conf.depth_radius(), 1);
    CHECK_LE_OR_RETURN(conf.depth_radius(), 16);
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

void LocalResponseNormalizationOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  in_shape.ToProto(kernel_conf->mutable_local_response_normalization_conf()->mutable_batch());
}

REGISTER_OP(OperatorConf::kLocalResponseNormalizationConf, LocalResponseNormalizationOp);

}  // namespace oneflow
