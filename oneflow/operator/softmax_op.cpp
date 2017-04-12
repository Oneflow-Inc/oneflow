#include "operator/softmax_op.h"
#include "glog/logging.h"

namespace oneflow {

void SoftmaxOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_softmax_op_conf());
  auto cnf = new SoftmaxOpConf(op_conf.softmax_op_conf());
  mut_pb_op_conf().reset(cnf);

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

} // namespace oneflow
