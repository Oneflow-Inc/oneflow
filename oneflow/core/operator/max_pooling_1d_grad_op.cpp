#include "oneflow/core/operator/max_pooling_1d_grad_op.h"

namespace oneflow {

const PbMessage& MaxPooling1DGradOp::GetCustomizedConf() const {
  return op_conf().max_pooling_1d_grad_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling1DGradConf, MaxPooling1DGradOp);

}  //  namespace oneflow
