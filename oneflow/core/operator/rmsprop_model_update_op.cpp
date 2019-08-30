#include "oneflow/core/operator/rmsprop_model_update_op.h"

namespace oneflow {

void RMSPropModelUpdateOp::MdUpdtVirtualInitFromOpConf() { EnrollTmpBn("mean_square"); }

Maybe<void> RMSPropModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ_OR_RETURN(model_blob_desc->data_type(), GlobalJobDesc().DefaultDataType());
  CHECK_EQ_OR_RETURN(model_blob_desc->has_data_id_field(), false);
  *GetBlobDesc4BnInOp("mean_square") = *model_blob_desc;
  return Maybe<void>::Ok();
}

const PbMessage& RMSPropModelUpdateOp::GetCustomizedConf() const {
  return op_conf().rmsprop_model_update_conf();
}

const HashSet<std::string> RMSPropModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kRmspropConf, NormalModelUpdtOp, RMSPropModelUpdateOp);

REGISTER_OP(OperatorConf::kRmspropModelUpdateConf, RMSPropModelUpdateOp);

}  // namespace oneflow
