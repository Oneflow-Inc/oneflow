#include "oneflow/core/operator/average_pooling_3d_op.h"

namespace oneflow {

const PbMessage& AveragePooling3DOp::GetCustomizedConf() const {
  return op_conf().average_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kAveragePooling3DConf, AveragePooling3DOp);

}  // namespace oneflow
