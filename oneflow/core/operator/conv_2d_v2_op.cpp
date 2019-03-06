#include "oneflow/core/operator/conv_2d_v2_op.h"

namespace oneflow {

const PbMessage& Conv2DV2Op::GetCustomizedConf() const { return op_conf().conv_2d_v2_conf(); }

REGISTER_OP(OperatorConf::kConv2DV2Conf, Conv2DV2Op);

}  // namespace oneflow
