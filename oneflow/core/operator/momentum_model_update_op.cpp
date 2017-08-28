#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void MomentumModelUpdateOp::InitFromOpConf() {
  CHECK(op_conf().has_momentum_mdupdt_conf());

  EnrollInputBn("model_diffs", false);
  EnrollDataTmpBn("momentum");
  EnrollOutputBn("model", false);
}

const PbMessage& MomentumModelUpdateOp::GetSpecialConf() const {
  return op_conf().momentum_mdupdt_conf();
}

void MomentumModelUpdateOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const BlobDesc* md_diff_blob_desc = GetBlobDesc4BnInOp("model_diffs");
  CHECK_EQ(md_diff_blob_desc->data_type(),
           JobDesc::Singleton()->default_data_type());
  CHECK_EQ(md_diff_blob_desc->has_data_id(), false);

  // momentum
  BlobDesc* momentum_blob_desc = GetBlobDesc4BnInOp("momentum");
  momentum_blob_desc->mut_shape() = md_diff_blob_desc->shape();
  momentum_blob_desc->set_data_type(md_diff_blob_desc->data_type());
  momentum_blob_desc->set_has_data_id(false);
}

REGISTER_OP(OperatorConf::kMomentumMdupdtConf, MomentumModelUpdateOp);

}  // namespace oneflow
