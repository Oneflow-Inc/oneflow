#include "oneflow/core/operator/deconv_2d_op.h"

namespace oneflow {

const PbMessage& Deconv2DOp::GetCustomizedConf() const { return op_conf().deconv_2d_conf(); }

REGISTER_OP(OperatorConf::kDeconv2DConf, Deconv2DOp);

}  // namespace oneflow
