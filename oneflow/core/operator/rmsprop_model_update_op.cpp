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

void RMSPropModelUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  // model_diffs
  const BlobDesc* model_diffs_blob_desc = GetBlobDesc4BnInOp("model_diffs");
  CHECK_EQ(model_diffs_blob_desc->data_type(),
           JobDesc::Singleton()->DefaultDataType());
  CHECK_EQ(model_diffs_blob_desc->has_data_id(), false);
  // mean_square
  BlobDesc* mean_square_blob_desc = GetBlobDesc4BnInOp("mean_square");
  *mean_square_blob_desc = *model_diffs_blob_desc;
}

REGISTER_OP(OperatorConf::kRmspropMdupdtConf, RMSPropModelUpdateOp);

}  // namespace oneflow
