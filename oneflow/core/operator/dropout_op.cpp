#include "oneflow/core/operator/dropout_op.h"

namespace oneflow {

void DropoutOp::InitFromOpConf() {
  double keep_prob = op_conf().dropout_conf().keep_prob();
  CHECK_GE(keep_prob, 0);
  CHECK_LE(keep_prob, 1);
  EnrollInputBn("in");
  EnrollInputBn("out");
  EnrollDataTmpBn("mask");
}

const PbMessage& DropoutOp::GetCustomizedConf() const {
  return op_conf().dropout_conf();
}

void DropoutOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(op_conf().dropout_conf().noise_shape().dim_size(),
           GetBlobDesc4BnInOp("in")->shape().NumAxes());
  *GetBlobDesc4BnInOp("mask") = *GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

void DropoutOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  auto mut_dropout_conf = kernel_conf->mutable_dropout_conf();
  GetBlobDesc4BnInOp("in")->shape().ToProto(mut_dropout_conf->mutable_in());
  GetBlobDesc4BnInOp("mask")->shape().ToProto(mut_dropout_conf->mutable_mask());
  GetBlobDesc4BnInOp("out")->shape().ToProto(mut_dropout_conf->mutable_out());
}

REGISTER_OP(OperatorConf::kDropoutConf, DropoutOp);

}  // namespace oneflow
