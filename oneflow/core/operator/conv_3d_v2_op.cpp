#include "oneflow/core/operator/conv_3d_v2_op.h"

namespace oneflow {

const PbMessage& Conv3DV2Op::GetCustomizedConf() const { return op_conf().conv_3d_v2_conf(); }

REGISTER_OP(OperatorConf::kConv3DV2Conf, Conv3DV2Op);

}  // namespace oneflow
