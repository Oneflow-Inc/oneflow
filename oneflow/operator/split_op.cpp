#include "operator/split_op.h"

namespace oneflow {

void SplitOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_split_op_conf());
  auto cnf = new SplitOpConf(op_conf.split_op_conf());
  mut_pb_op_conf().reset(cnf);
}

std::string SplitOp::ibn2lbn(const std::string& input_blob_name) const {
  UNEXPECTED_RUN();
}

std::string SplitOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  UNEXPECTED_RUN();
}

std::string SplitOp::obn2lbn(const std::string& output_blob_name) const {
  UNEXPECTED_RUN();
}

std::string SplitOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  UNEXPECTED_RUN();
}

} // namespace oneflow
