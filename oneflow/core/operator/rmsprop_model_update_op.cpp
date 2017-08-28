#include "oneflow/core/operator/rmsprop_model_update_op.h"

namespace oneflow {

void RMSPropModelUpdateOp::InitFromOpConf() {
  EnrollInputBn("model_diffs", false);
  EnrollDataTmpBn("mean_square");
  EnrollOutputBn("model", false);
}

const PbMessage& RMSPropModelUpdateOp::GetSpecialConf() const {
  return op_conf().rmsprop_mdupdt_conf();
}

void RMSPropModelUpdateOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  TODO();
}

REGISTER_OP(OperatorConf::kRmspropMdupdtConf, RMSPropModelUpdateOp);

}  // namespace oneflow
