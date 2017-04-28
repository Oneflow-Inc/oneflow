#include "operator/clone_op.h"
#include "operator/operator_manager.h"

namespace oneflow {

void CloneOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_clone_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("in");
  for (int64_t i = 0; i < op_conf.clone_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}
std::string CloneOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().clone_conf(), k);
}

void CloneOp::InferShape4ObAndDtbFromIb() const {
  Shape* input_shape_ptr = GetShapePtr(SoleIbn());
  const std::vector<int64_t>& input_dim_vec = input_shape_ptr->dim_vec();
  for(std::string obn : output_bns()){
    *GetShapePtr(obn) = Shape(input_dim_vec);
  }
}

REGISTER_OP(OperatorConf::kCloneConf, CloneOp);

} // namespace oneflow
