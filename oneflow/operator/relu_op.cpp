#include "operator/relu_op.h"
#include "glog/logging.h"

namespace oneflow {

void ReluOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_relu_op_conf());
  auto cnf = new ReluOpConf(op_conf.relu_op_conf());
  mut_pb_op_conf().reset(cnf);

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

} // namespace oneflow
