#include "oneflow/core/operator/max_pooling_3d_op.h"

namespace oneflow {

const PbMessage& MaxPooling3DOp::GetCustomizedConf() const {
  return op_conf().max_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling3DConf, MaxPooling3DOp);

}  //  namespace oneflow
