#include "oneflow/core/operator/embedding_lookup_accumulate_op.h"

namespace oneflow {

void EmbeddingLookupAccumulateOp::InitFromOpConf() {
  CHECK(op_conf().has_accumulate_conf());

  EnrollInputBn("one_ids", false);
  EnrollInputBn("one_val", false);
  EnrollOutputBn("acc_ids", false);
  EnrollOutputBn("acc_val", false);
}

const PbMessage& EmbeddingLookupAccumulateOp::GetCustomizedConf() const {
  return op_conf().embedding_lookup_accumulate_conf();
}

REGISTER_OP(OperatorConf::kEmbeddingLookupAccumulateConf, EmbeddingLookupAccumulateOp);

}  // namespace oneflow
