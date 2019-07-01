#include "oneflow/core/operator/lars_model_update_op.h"

namespace oneflow {

void LARSModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    EnrollInputBn("momentum", false)->set_is_mutable(true);
  } else {
    UNIMPLEMENTED();
  }
  EnrollDataTmpBn("data_tmp");
}

void LARSModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    CHECK(*GetBlobDesc4BnInOp("momentum") == *model_blob_desc);
  } else {
    UNIMPLEMENTED();
  }

  // data_tmp for gpu compute
  // data_tmp[0] for model_norm, data_tmp[1] for model_diff_norm, data_tmp[2] for
  // local_learning_rate
  *GetBlobDesc4BnInOp("data_tmp") = *model_blob_desc;
  GetBlobDesc4BnInOp("data_tmp")->mut_shape() = Shape({3});
}

const PbMessage& LARSModelUpdateOp::GetCustomizedConf() const {
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return op_conf().lars_model_update_conf();
  } else {
    UNIMPLEMENTED();
  }
}

const HashSet<std::string> LARSModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{"data_tmp"};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kLarsConf, NormalModelUpdtOp, LARSModelUpdateOp);

REGISTER_OP(OperatorConf::kLarsModelUpdateConf, LARSModelUpdateOp);

}  // namespace oneflow
