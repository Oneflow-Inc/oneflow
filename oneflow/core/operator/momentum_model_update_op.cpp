#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void MomentumModelUpdateOp::MdUpdtVirtualInitFromOpConf() { EnrollForwardModelBn("momentum"); }

void MomentumModelUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ(model_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(model_blob_desc->has_data_id_field(), false);
  *GetBlobDesc4BnInOp("momentum") = *model_blob_desc;
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kMomentumConf, NormalModelUpdtOp,
               MomentumModelUpdateOp);

}  // namespace oneflow
