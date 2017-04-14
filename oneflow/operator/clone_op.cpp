#include "operator/clone_op.h"

namespace oneflow {

void CloneOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_clone_op_conf());
  auto cnf = new CloneOpConf(op_conf.clone_op_conf());
  mut_pb_op_conf().reset(cnf);

  is_boxing_ = cnf->is_boxing();

  EnrollInputBn("in");
  for (int32_t i = 0; i < cnf->out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

} // namespace oneflow
