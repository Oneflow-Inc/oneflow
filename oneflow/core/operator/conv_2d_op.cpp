#include "oneflow/core/operator/conv_2d_op.h"

namespace oneflow {

const PbMessage& Conv2DOp::GetCustomizedConf() const {
  return op_conf().conv_2d_conf();
}

ActivationType Conv2DOp::GetActivationType() const {
  CHECK(op_conf().conv_2d_conf().has_activation());
  return op_conf().conv_2d_conf().activation();
}

bool Conv2DOp::UseActivation() const {
  return op_conf().conv_2d_conf().has_activation();
}

REGISTER_OP(OperatorConf::kConv2DConf, Conv2DOp);

}  // namespace oneflow
