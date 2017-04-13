#include "operator/split_op.h"

namespace oneflow {

void SplitOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_split_op_conf());
  auto cnf = new SplitOpConf(op_conf.split_op_conf());
  mut_pb_op_conf().reset(cnf);
  
  EnrollInputBn("in");
  for (int32_t i = 0; i < cnf->out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

} // namespace oneflow
