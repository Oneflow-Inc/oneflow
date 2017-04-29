#include "operator/softmax_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_softmax_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

std::string SoftmaxOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().softmax_conf(), k);
}
void SoftmaxOp::InferShape4ObAndDtbFromIb() const {
  std::vector<int64_t> vec = GetShapePtr(SoleIbn())->dim_vec();
  CHECK_GT(vec.size(), 1);
  vec.erase(vec.begin() + op_conf().softmax_conf().axis());
  *GetShapePtr(SoleObn()) = Shape(vec);
}
REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

} // namespace oneflow
