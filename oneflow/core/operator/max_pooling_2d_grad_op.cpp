#include "oneflow/core/operator/max_pooling_2d_grad_op.h"

namespace oneflow {

const PbMessage& MaxPooling2DGradOp::GetCustomizedConf() const {
  return op_conf().max_pooling_2d_grad_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling2DGradConf, MaxPooling2DGradOp);

}  //  namespace oneflow
