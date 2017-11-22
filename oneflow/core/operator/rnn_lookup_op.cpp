#include "oneflow/core/operator/rnn_lookup_op.h"

namespace oneflow {

void RnnLookupOp::InitFromOpConf() {
  CHECK(op_conf().has_rnn_lookup_conf());

  EnrollInputBn("in", false);
  EnrollOutputBn("out");
  EnrollModelBn("weight");
}

const PbMessage& RnnLookupOp::GetSpecialConf() const {
  return op_conf().rnn_lookup_conf();
}

void RnnLookupOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  TODO();
}

REGISTER_OP(OperatorConf::kRnnLookupConf, RnnLookupOp);

}  // namespace oneflow
