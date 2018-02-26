#include "oneflow/core/operator/max_pooling_2d_op.h"

namespace oneflow {

const PbMessage& MaxPooling2DOp::GetSpecialConf() const {
  return op_conf().max_pooling_2d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling2DConf, MaxPooling2DOp);

}  //  namespace oneflow
