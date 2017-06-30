#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

void CloneOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_clone_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  for (int64_t i = 0; i < op_conf.clone_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

const PbMessage& CloneOp::GetSpecialConf() const {
  return op_conf().clone_conf();
}

void CloneOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  for (std::string obn : output_bns()) {
    *GetShapePtr4BnInOp(obn) = *input_shape_ptr;
  }
}

REGISTER_OP(OperatorConf::kCloneConf, CloneOp);

}  // namespace oneflow
