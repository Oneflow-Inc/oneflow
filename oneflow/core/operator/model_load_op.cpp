#include "oneflow/core/operator/model_load_op.h"

namespace oneflow {

void ModelLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_model_load_conf());
  EnrollInputBn("out", false);
}

const PbMessage& ModelLoadOp::GetCustomizedConf() const { return op_conf().model_load_conf(); }

REGISTER_OP(OperatorConf::kModelLoadConf, ModelLoadOp);

}  // namespace oneflow
