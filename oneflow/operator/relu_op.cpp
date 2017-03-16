#include "operator/relu_op.h"
#include "glog/logging.h"

namespace oneflow {

void ReluOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_relu_op_conf());
  auto cnf = new ReluOpConf(op_conf.relu_op_conf());
  mutable_pb_op_conf().reset(cnf);

  RegisterInputBlobName("in");
  RegisterInputDiffBlobName(GenDiffBlobName("in"));
  RegisterOutputBlobName("out");
  RegisterOutputDiffBlobName(GenDiffBlobName("out"));
}

} // namespace oneflow
