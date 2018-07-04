#include "oneflow/core/operator/conv_1d_op.h"

namespace oneflow {

const PbMessage& Conv1DOp::GetCustomizedConf() const { return op_conf().conv_1d_conf(); }

REGISTER_OP(OperatorConf::kConv1DConf, Conv1DOp);

}  // namespace oneflow
