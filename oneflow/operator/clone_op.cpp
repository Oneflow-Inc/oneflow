#include "operator/clone_op.h"

namespace oneflow {

void CloneOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_clone_op_conf());
  auto cnf = new CloneOpConf(op_conf.clone_op_conf());
  mut_pb_op_conf().reset(cnf);
}

std::string CloneOp::ibn2lbn(const std::string& input_blob_name) const {
  UNEXPECTED_RUN();
}

std::string CloneOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  UNEXPECTED_RUN();
}

std::string CloneOp::obn2lbn(const std::string& output_blob_name) const {
  UNEXPECTED_RUN();
}

std::string CloneOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  UNEXPECTED_RUN();
}

} // namespace oneflow
