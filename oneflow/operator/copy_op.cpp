#include "operator/copy_op.h"

namespace oneflow {

void CopyOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_copy_op_conf());
  auto cnf = new CopyOpConf(op_conf.copy_op_conf());
  mut_pb_op_conf().reset(cnf);

  for (int32_t i = 0; i < cnf->copied_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i));
    EnrollOutputBn("out_" + std::to_string(i));
  }

}

} // namespace oneflow
