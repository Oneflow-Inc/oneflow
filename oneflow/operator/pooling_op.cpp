#include "operator/pooling_op.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

void PoolingOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_pooling_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("idx");
}

std::string PoolingOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().pooling_conf(), k);
}

REGISTER_OP(OperatorConf::kPoolingConf, PoolingOp);

} // namespace oneflow
