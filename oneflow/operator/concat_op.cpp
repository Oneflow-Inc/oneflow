#include "operator/concat_op.h"

namespace oneflow {

void ConcatOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_concat_op_conf());
  auto cnf = new ConcatOpConf(op_conf.concat_op_conf());
  mut_pb_op_conf().reset(cnf);
}

} // namespace oneflow
