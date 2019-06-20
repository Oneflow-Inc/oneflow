#include "oneflow/core/operator/rmsprop_model_update_op.h"

namespace oneflow {

void RMSPropModelUpdateOp::MdUpdtVirtualInitFromOpConf() { EnrollDataTmpBn("mean_square"); }

void RMSPropModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ(model_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(model_blob_desc->has_data_id_field(), false);
  *GetBlobDesc4BnInOp("mean_square") = *model_blob_desc;
}

const PbMessage& RMSPropModelUpdateOp::GetCustomizedConf() const {
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return op_conf().rmsprop_model_update_conf();
  } else {
    UNIMPLEMENTED();
  }
}

const HashSet<std::string> RMSPropModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kRmspropConf, NormalModelUpdtOp, RMSPropModelUpdateOp);

REGISTER_OP(OperatorConf::kRmspropModelUpdateConf, RMSPropModelUpdateOp);

}  // namespace oneflow
