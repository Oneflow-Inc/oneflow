#include "oneflow/core/operator/average_pooling_2d_op.h"

namespace oneflow {

const PbMessage& AveragePooling2DOp::GetCustomizedConf() const {
  return op_conf().average_pooling_2d_conf();
}

REGISTER_OP(OperatorConf::kAveragePooling2DConf, AveragePooling2DOp);

}  // namespace oneflow
