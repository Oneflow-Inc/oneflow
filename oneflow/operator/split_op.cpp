#include "operator/split_op.h"

namespace oneflow {

void SplitOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_split_op_conf());
  auto cnf = new SplitOpConf(op_conf.split_op_conf());
  mut_pb_op_conf().reset(cnf);
}

} // namespace oneflow
