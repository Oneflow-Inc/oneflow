#include "oneflow/core/operator/relu_op.h"

namespace oneflow {

void ReluOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_relu_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ReluOp::GetSpecialConf() const {
  return op_conf().relu_conf();
}

void ReluOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  Shape* output_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  *output_shape_ptr = *input_shape_ptr;
}

REGISTER_OP(OperatorConf::kReluConf, ReluOp);

}  // namespace oneflow
