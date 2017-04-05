#include "operator/clone_op.h"

namespace oneflow {

void CloneOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_clone_op_conf());
  auto cnf = new CloneOpConf(op_conf.clone_op_conf());
  mut_pb_op_conf().reset(cnf);
}

} // namespace oneflow
