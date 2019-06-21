#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void MomentumModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  EnrollInputBn("momentum", false)->set_is_mutable(true);
}

void MomentumModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ(model_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(model_blob_desc->has_data_id_field(), false);
  CHECK(*GetBlobDesc4BnInOp("momentum") == *model_blob_desc);
}

const PbMessage& MomentumModelUpdateOp::GetCustomizedConf() const {
  CHECK(Global<JobDesc>::Get()->IsTrain());
  return op_conf().momentum_model_update_conf();
}

const HashSet<std::string> MomentumModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kMomentumConf, NormalModelUpdtOp,
               MomentumModelUpdateOp);

REGISTER_OP(OperatorConf::kMomentumModelUpdateConf, MomentumModelUpdateOp);

}  // namespace oneflow
