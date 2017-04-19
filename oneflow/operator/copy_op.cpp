#include "operator/copy_op.h"

namespace oneflow {

void CopyOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_copy_op_conf());
  auto cnf = new CopyOpConf(op_conf.copy_op_conf());
  mut_pb_op_conf().reset(cnf);

  for (int64_t i = 0; i < cnf->copied_lbns_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    EnrollInputBn(ibn);
    CHECK(ibn2lbn_.emplace(ibn, cnf->copied_lbns(i)).second);
    std::string obn = "out_" + std::to_string(i);
    EnrollOutputBn(obn);
    CHECK(obn2lbn_.emplace(obn, cnf->copied_lbns(i)).second);
  }
}

} // namespace oneflow
