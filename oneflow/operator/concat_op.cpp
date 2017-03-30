#include "operator/concat_op.h"

namespace oneflow {

void ConcatOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_concat_op_conf());
  auto cnf = new ConcatOpConf(op_conf.concat_op_conf());
  mut_pb_op_conf().reset(cnf);
}

std::string ConcatOp::ibn2lbn(const std::string& input_blob_name) const {
  UNEXPECTED_RUN();
}

std::string ConcatOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  UNEXPECTED_RUN();
}

std::string ConcatOp::obn2lbn(const std::string& output_blob_name) const {
  UNEXPECTED_RUN();
}

std::string ConcatOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  UNEXPECTED_RUN();
}

} // namespace oneflow
