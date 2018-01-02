#include "oneflow/core/operator/rmsprop_model_update_op.h"

namespace oneflow {

void RMSPropModelUpdateOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_rmsprop_mdupdt_conf());
  EnrollDataTmpBn("mean_square");
}

const PbMessage& RMSPropModelUpdateOp::GetSpecialConf() const {
  return op_conf().rmsprop_mdupdt_conf();
}

void RMSPropModelUpdateOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  *GetBlobDesc4BnInOp("mean_square") = *model_blob_desc;
}

REGISTER_OP(OperatorConf::kRmspropMdupdtConf, RMSPropModelUpdateOp);

}  // namespace oneflow
