#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void MomentumModelUpdateOp::InitFromOpConf() {
  CHECK(op_conf().has_momentum_mdupdt_conf());

  EnrollInputBn("model_diff_acc", false);
  EnrollDataTmpBn("momentum");
  EnrollOutputBn("model", false);
}

const PbMessage& MomentumModelUpdateOp::GetSpecialConf() const {
  return op_conf().momentum_mdupdt_conf();
}

void MomentumModelUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ(model_blob_desc->data_type(),
           JobDesc::Singleton()->default_data_type());
  CHECK_EQ(model_blob_desc->has_data_id(), false);
  *GetBlobDesc4BnInOp("momentum") = *model_blob_desc;
}

REGISTER_OP(OperatorConf::kMomentumMdupdtConf, MomentumModelUpdateOp);

}  // namespace oneflow
