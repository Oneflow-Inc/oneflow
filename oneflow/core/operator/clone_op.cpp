#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

void CloneOp::InitFromOpConf() {
  EnrollInputBn("in");
  for (int64_t i = 0; i < op_conf().clone_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

const PbMessage& CloneOp::GetSpecialConf() const {
  return op_conf().clone_conf();
}

void CloneOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  const BlobDesc* input_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  for (std::string obn : output_bns()) {
    *GetBlobDesc4BnInOp(obn) = *input_blob_desc;
  }
}

void CloneOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext*, KernelConf* kernel_conf) const {
  DataType dtype = GetBlobDesc4BnInOp("in")->data_type();
  kernel_conf->mutable_clone_conf()->set_data_type(dtype);
}

REGISTER_OP(OperatorConf::kCloneConf, CloneOp);

}  // namespace oneflow
