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

void SoftmaxOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  std::vector<int64_t> vec = GetShapePtr4BnInOp(SoleIbn())->dim_vec();
  CHECK_GT(vec.size(), 1);
  int32_t axis = (op_conf().softmax_conf().axis() + vec.size()) % vec.size();
  vec.erase(vec.begin() + axis);
  *GetShapePtr4BnInOp(SoleObn()) = Shape(vec);
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

} // namespace oneflow
