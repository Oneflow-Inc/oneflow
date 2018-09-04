#include "oneflow/core/operator/naive_model_update_op.h"
#include "oneflow/core/operator/rmsprop_model_update_op.h"
#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void NormalModelUpdtOp::InitFromOpConf() {
  // TODO(jiyuan): single in for mdupdt op
  FOR_RANGE(int32_t, i, 0, op_conf().normal_mdupdt_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("model", false);
  MdUpdtVirtualInitFromOpConf();
}

const PbMessage& NormalModelUpdtOp::GetCustomizedConf() const {
  return op_conf().normal_mdupdt_conf();
}

REGISTER_OP_CREATOR(OperatorConf::kNormalMdupdtConf, [](const OperatorConf& op_conf) -> Operator* {
  return NewObj<NormalModelUpdtOp>(op_conf.normal_mdupdt_conf().user_conf().normal_mdupdt_case());
});

}  // namespace oneflow
