#include "oneflow/core/operator/lars_model_update_op.h"

namespace oneflow {

void LARSModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  EnrollInputBn("momentum", false)->set_is_mutable(true);
  EnrollTmpBn("data_tmp");
}

Maybe<void> LARSModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("momentum") == *model_blob_desc);

  // data_tmp for gpu compute
  // data_tmp[0] for model_norm, data_tmp[1] for model_diff_norm, data_tmp[2] for
  // local_learning_rate
  *GetBlobDesc4BnInOp("data_tmp") = *model_blob_desc;
  GetBlobDesc4BnInOp("data_tmp")->mut_shape() = Shape({3});
  return Maybe<void>::Ok();
}

const PbMessage& LARSModelUpdateOp::GetCustomizedConf() const {
  return op_conf().lars_model_update_conf();
}

const HashSet<std::string> LARSModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{"data_tmp"};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kLarsConf, NormalModelUpdtOp, LARSModelUpdateOp);

REGISTER_OP(OperatorConf::kLarsModelUpdateConf, LARSModelUpdateOp);

}  // namespace oneflow
