#include "oneflow/core/operator/lars_model_update_op.h"

namespace oneflow {

void LARSModelUpdateOp::InitFromOpConf() {
  CHECK(op_conf().has_lars_mdupdt_conf());

  EnrollInputBn("model_diffs", false);
  EnrollDataTmpBn("momentum");
  EnrollDataTmpBn("temp");
  EnrollOutputBn("model", false);
}

const PbMessage& LARSModelUpdateOp::GetSpecialConf() const {
  return op_conf().lars_mdupdt_conf();
}

void LARSModelUpdateOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const BlobDesc* md_diff_blob_desc = GetBlobDesc4BnInOp("model_diffs");
  CHECK_EQ(md_diff_blob_desc->data_type(),
           JobDesc::Singleton()->default_data_type());
  CHECK_EQ(md_diff_blob_desc->has_data_id(), false);

  // momentum
  *GetBlobDesc4BnInOp("momentum") = *md_diff_blob_desc;

  // temp
  BlobDesc* temp_blob_desc = GetBlobDesc4BnInOp("temp");
  temp_blob_desc->mut_shape() = Shape({md_diff_blob_desc->shape().At(0) * 2});
  temp_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  temp_blob_desc->set_has_data_id(false);
}

REGISTER_OP(OperatorConf::kLarsMdupdtConf, LARSModelUpdateOp);

}  // namespace oneflow
