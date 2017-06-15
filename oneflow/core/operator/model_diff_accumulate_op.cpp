#include "oneflow/core/operator/model_diff_accumulate_op.h"
#include "glog/logging.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void ModelDiffAccOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_diff_acc_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("model_diff", false);
  EnrollOutputBn("model_diff_acc", false);
}

const PbMessage& ModelDiffAccOp::GetSpecialConf() const {
  return op_conf().model_update_conf();
}

REGISTER_OP(OperatorConf::kModelDiffAccConf, ModelDiffAccOp);

}
