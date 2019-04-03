#include "oneflow/core/operator/model_save_v2_op.h"

namespace oneflow {

void ModelSaveV2Op::InitFromOpConf() {
  CHECK(op_conf().has_model_save_v2_conf());
  EnrollInputBn("in", false);
}

const PbMessage& ModelSaveV2Op::GetCustomizedConf() const { return op_conf().model_save_v2_conf(); }

REGISTER_OP(OperatorConf::kModelSaveV2Conf, ModelSaveV2Op);

}  // namespace oneflow
