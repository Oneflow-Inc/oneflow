#include "oneflow/core/operator/average_pooling_op.h"

namespace oneflow {

const PbMessage& AveragePoolingOp::GetSpecialConf() const {
  return op_conf().average_pooling_conf();
}

REGISTER_OP(OperatorConf::kAveragePoolingConf, AveragePoolingOp);

}  // namespace oneflow
