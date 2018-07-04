#include "oneflow/core/operator/conv_3d_op.h"

namespace oneflow {

const PbMessage& Conv3DOp::GetCustomizedConf() const { return op_conf().conv_3d_conf(); }

REGISTER_OP(OperatorConf::kConv3DConf, Conv3DOp);

}  // namespace oneflow
