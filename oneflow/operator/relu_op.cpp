#include "operator/relu_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void ReluOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_relu_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

std::string ReluOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().relu_conf(), k);
}

REGISTER_OP(OperatorConf::kReluConf, ReluOp);

void ReluOp::InferShape4ObAndDtbFromIb() const {
  Shape* output_shape_ptr = GetShapePtr(SoleObn());
  Shape* input_shape_ptr = GetShapePtr(SoleIbn());
  const std::vector<int64_t>& input_shape_dim_vec = input_shape_ptr->dim_vec();
  *output_shape_ptr = Shape(input_shape_dim_vec);
}

} // namespace oneflow
