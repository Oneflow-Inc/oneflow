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

void CloneOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const BlobDesc* input_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  for (std::string obn : output_bns()) {
    *GetBlobDesc4BnInOp(obn) = *input_blob_desc;
  }
  mut_op_conf().mutable_clone_conf()->set_data_type(
      input_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kCloneConf, CloneOp);

}  // namespace oneflow
