#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void MomentumModelUpdateOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_momentum_mdupdt_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("model_diffs", false);
  EnrollModelTmpBn("momentum");
  EnrollOutputBn("model", false);
}

const PbMessage& MomentumModelUpdateOp::GetSpecialConf() const {
  return op_conf().momentum_mdupdt_conf();
}

void MomentumModelUpdateOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  *GetShapePtr4BnInOp("momentum") = *input_shape_ptr;
}

REGISTER_OP(OperatorConf::kMomentumMdupdtConf, MomentumModelUpdateOp);

}  // namespace oneflow
