#include "operator/softmax_op.h"
#include "glog/logging.h"

namespace oneflow {

void SoftmaxOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_softmax_op_conf());
  auto cnf = new SoftmaxOpConf(op_conf.softmax_op_conf());
  mutable_pb_op_conf().reset(cnf);

  RegisterInputBlobName("in");
  RegisterInputDiffBlobName(GenDiffBlobName("in"));
  RegisterOutputBlobName("out");
  RegisterOutputDiffBlobName(GenDiffBlobName("out"));
}

} // namespace oneflow
