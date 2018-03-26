#include "oneflow/core/operator/local_reponse_normalization_op.h"

namespace oneflow {

void LocalResponseNormalizationOp::InitFromOpConf() {
  CHECK(op_conf().has_local_response_normalization_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("padded_square");
  EnrollDataTmpBn("normalize_coef");
}

const PbMessage& LocalResponseNormalizationOp::GetCustomizedConf() const {
  return op_conf().local_response_normalization_conf();
}

void LocalResponseNormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type) const {
  const LocalResponseNormalizationOpConf conf =
      op_conf().local_response_normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(),
           Global<JobDesc>::Get()->DefaultDataType());
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;

  if (device_type == DeviceType::kCPU) {
    GetBlobDesc4BnInOp("padded_square")->mut_shape() =
        Shape({in_blob_desc->shape().At(3) + 2 * conf.depth_radius()});
    GetBlobDesc4BnInOp("normalize_coef")->mut_shape() = in_blob_desc->shape();
  } else if (device_type == DeviceType::kGPU) {
    // cudnn requirements
    CHECK_GE(conf.bias(), 1e-5);
    CHECK_GE(conf.beta(), 0.01);
    CHECK_GE(conf.depth_radius(), 1);
    CHECK_LE(conf.depth_radius(), 16);
  } else {
    UNIMPLEMENTED();
  }
}

void LocalResponseNormalizationOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  in_shape.ToProto(kernel_conf->mutable_local_response_normalization_conf()
                       ->mutable_batch());
}

}  // namespace oneflow
