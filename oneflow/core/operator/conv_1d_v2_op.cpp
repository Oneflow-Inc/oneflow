#include "oneflow/core/operator/conv_1d_v2_op.h"

namespace oneflow {

const PbMessage& Conv1DV2Op::GetCustomizedConf() const { return op_conf().conv_1d_v2_conf(); }

REGISTER_OP(OperatorConf::kConv1DV2Conf, Conv1DV2Op);

}  // namespace oneflow
