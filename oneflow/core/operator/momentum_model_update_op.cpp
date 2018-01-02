#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void MomentumModelUpdateOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_momentum_mdupdt_conf());
  EnrollDataTmpBn("momentum");
}

const PbMessage& MomentumModelUpdateOp::GetSpecialConf() const {
  return op_conf().momentum_mdupdt_conf();
}

void MomentumModelUpdateOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  *GetBlobDesc4BnInOp("momentum") = *model_blob_desc;
}

REGISTER_OP(OperatorConf::kMomentumMdupdtConf, MomentumModelUpdateOp);

}  // namespace oneflow
