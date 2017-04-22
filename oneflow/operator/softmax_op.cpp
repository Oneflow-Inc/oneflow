#include "operator/softmax_op.h"
#include "glog/logging.h"

namespace oneflow {

<<<<<<< HEAD
void SoftmaxOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_softmax_op_conf());
  mut_op_conf() = op_conf;
=======
void SoftmaxOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_softmax_conf());
  auto cnf = new SoftmaxOpConf(op_conf.softmax_conf());
  mut_pb_op_conf().reset(cnf);
>>>>>>> 290256f7ca40d3ca9a1864d713fc5d78422a2cce

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

std::string SoftmaxOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().softmax_op_conf(), k);
}
} // namespace oneflow
