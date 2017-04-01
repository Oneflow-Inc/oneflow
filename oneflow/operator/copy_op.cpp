#include "operator/copy_op.h"

namespace oneflow {

void CopyOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_copy_op_conf());
  auto cnf = new CopyOpConf(op_conf.copy_op_conf());
  mut_pb_op_conf().reset(cnf);
}

std::string CopyOp::ibn2lbn(const std::string& input_blob_name) const {
  UNEXPECTED_RUN();
}
std::string CopyOp::obn2lbn(const std::string& output_blob_name) const {
  UNEXPECTED_RUN();
}

std::string CopyOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  UNEXPECTED_RUN();
}

std::string CopyOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  UNEXPECTED_RUN();
}

} // namespace oneflow
