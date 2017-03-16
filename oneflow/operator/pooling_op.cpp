#include "operator/pooling_op.h"
#include "glog/logging.h"

namespace oneflow {

void PoolingOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_pooling_op_conf());
  auto cnf = new PoolingOpConf(op_conf.pooling_op_conf());
  mutable_pb_op_conf().reset(cnf);

  RegisterInputBlobName("in");
  RegisterInputDiffBlobName(GenDiffBlobName("in"));
  RegisterOutputBlobName("out");
  RegisterOutputDiffBlobName(GenDiffBlobName("out"));
  RegisterDataTmpBlobName("idx");
}

} // namespace oneflow
