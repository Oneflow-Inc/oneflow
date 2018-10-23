#include "oneflow/core/operator/deconv_3d_op.h"

namespace oneflow {

const PbMessage& Deconv3DOp::GetCustomizedConf() const { return op_conf().deconv_3d_conf(); }

REGISTER_OP(OperatorConf::kDeconv3DConf, Deconv3DOp);

}  // namespace oneflow
