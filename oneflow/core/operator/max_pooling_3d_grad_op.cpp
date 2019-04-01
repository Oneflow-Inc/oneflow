#include "oneflow/core/operator/max_pooling_3d_grad_op.h"

namespace oneflow {

const PbMessage& MaxPooling3DGradOp::GetCustomizedConf() const {
  return op_conf().max_pooling_3d_grad_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling3DGradConf, MaxPooling3DGradOp);

}  //  namespace oneflow
