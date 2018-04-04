#include "oneflow/core/operator/conv_1d_op.h"

namespace oneflow {

const PbMessage& Conv1DOp::GetCustomizedConf() const {
  return op_conf().conv_1d_conf();
}

ActivationType Conv1DOp::GetActivationType() const {
  CHECK(op_conf().conv_1d_conf().has_activation());
  return op_conf().conv_1d_conf().activation();
}

REGISTER_OP(OperatorConf::kConv1DConf, Conv1DOp);

}  // namespace oneflow
