#include "oneflow/core/operator/dropout_op.h"

namespace oneflow {

void DropoutOp::InitFromOpConf() {
  if (op_conf().dropout_conf().has_noise_shape()) { TODO(); }
  if (op_conf().dropout_conf().has_seed()) { TODO(); }
  double keep_prob = op_conf().dropout_conf().keep_prob();
  CHECK_GT(keep_prob, 0);
  CHECK_LE(keep_prob, 1);
  EnrollInputBn("in");
  EnrollInputBn("out");
  if (JobDesc::Singleton()->IsTrain()) { EnrollDataTmpBn("mask"); }
}

const PbMessage& DropoutOp::GetCustomizedConf() const {
  return op_conf().dropout_conf();
}

void DropoutOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(op_conf().dropout_conf().noise_shape().dim_size(),
           GetBlobDesc4BnInOp("in")->shape().NumAxes());
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  if (JobDesc::Singleton()->IsTrain()) {
    *GetBlobDesc4BnInOp("mask") = *GetBlobDesc4BnInOp("in");
    GetBlobDesc4BnInOp("mask")->set_data_type(DataType::kFloat);
  }
}

void DropoutOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  DropoutKernelConf* mut_dropout_conf = kernel_conf->mutable_dropout_conf();
  GetBlobDesc4BnInOp("in")->shape().ToProto(mut_dropout_conf->mutable_in());
  GetBlobDesc4BnInOp("in")->shape().ToProto(mut_dropout_conf->mutable_mask());
  GetBlobDesc4BnInOp("out")->shape().ToProto(mut_dropout_conf->mutable_out());
}

REGISTER_OP(OperatorConf::kDropoutConf, DropoutOp);

}  // namespace oneflow
