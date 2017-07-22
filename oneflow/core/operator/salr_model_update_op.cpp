#include "oneflow/core/operator/salr_model_update_op.h"

namespace oneflow {

void SALRModelUpdateOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_salr_mdupdt_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("model_diff", false);
  EnrollModelTmpBn("learning_rate");
  EnrollOutputBn("model", false);
  EnrollDataTmpBn("last_diffs_flag");
}

void SALRModelUpdateOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  *GetShapePtr4BnInOp("learning_rate") = *input_shape_ptr;
  *GetShapePtr4BnInOp("last_diffs_flag") = *input_shape_ptr;
}

const PbMessage& SALRModelUpdateOp::GetSpecialConf() const {
  return op_conf().salr_mdupdt_conf();
}

REGISTER_OP(OperatorConf::kSalrMdupdtConf, SALRModelUpdateOp);

}  // namespace oneflow
