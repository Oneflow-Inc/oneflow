#include "oneflow/core/operator/normal_model_update_op.h"
#include "oneflow/core/operator/rmsprop_model_update_op.h"
#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void ModelUpdtOp::InitFromOpConf() {
  FOR_RANGE(int32_t, i, 0, op_conf().mdupdt_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("model", false);
  MdUpdtVirtualInitFromOpConf();
}

REGISTER_OP_CREATOR(OperatorConf::kMdupdtConf,
                    [](const OperatorConf& op_conf) -> Operator* {
                      const ModelUpdateOpUserConf& user_conf =
                          op_conf.mdupdt_conf().user_conf();
                      if (user_conf.has_normal_conf()) {
                        return new NormalModelUpdateOp;
                      } else if (user_conf.has_momentum_conf()) {
                        return new MomentumModelUpdateOp;
                      } else if (user_conf.has_rmsprop_conf()) {
                        return new RMSPropModelUpdateOp;
                      } else {
                        UNIMPLEMENTED();
                      }
                    });

}  // namespace oneflow
