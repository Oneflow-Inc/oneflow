#include "oneflow/core/operator/max_pooling_1d_op.h"

namespace oneflow {

const PbMessage& MaxPooling1DOp::GetCustomizedConf() const {
  return op_conf().max_pooling_1d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling1DConf, MaxPooling1DOp);

}  //  namespace oneflow
