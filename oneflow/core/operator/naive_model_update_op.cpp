#include "oneflow/core/operator/naive_model_update_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

const PbMessage& NaiveModelUpdateOp::GetCustomizedConf() const {
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return op_conf().naive_model_update_conf();
  } else {
    UNIMPLEMENTED();
  }
}

const HashSet<std::string> NaiveModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kNaiveConf, NormalModelUpdtOp, NaiveModelUpdateOp);

REGISTER_OP(OperatorConf::kNaiveModelUpdateConf, NaiveModelUpdateOp);

}  // namespace oneflow
