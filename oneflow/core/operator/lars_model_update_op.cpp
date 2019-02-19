#include "oneflow/core/operator/lars_model_update_op.h"

namespace oneflow {

void LARSModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  EnrollDataTmpBn("momentum");
  EnrollDataTmpBn("data_tmp");
}

void LARSModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ(model_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(model_blob_desc->has_data_id_field(), false);
  *GetBlobDesc4BnInOp("momentum") = *model_blob_desc;

  // data_tmp for gpu compute
  // data_tmp[0] for model_norm, data_tmp[1] for model_diff_norm, data_tmp[2] for
  // local_learning_rate
  *GetBlobDesc4BnInOp("data_tmp") = *model_blob_desc;
  GetBlobDesc4BnInOp("data_tmp")->mut_shape() = Shape({3});
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kLarsConf, NormalModelUpdtOp, LARSModelUpdateOp);

}  // namespace oneflow
