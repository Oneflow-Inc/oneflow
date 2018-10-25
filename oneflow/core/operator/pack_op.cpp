#include "oneflow/core/operator/pack_op.h"

namespace oneflow {

void PackOp::InitFromOpConf() {
  CHECK(op_conf().has_pack_conf());

  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

REGISTER_OP(OperatorConf::kPackConf, PackOp);

}  // namespace oneflow
