#include "operator/copy_op.h"

namespace oneflow {

void CopyOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_copy_op_conf());
  auto cnf = new CopyOpConf(op_conf.copy_op_conf());
  mut_pb_op_conf().reset(cnf);

  for (const std::string& lbn : cnf->lbns()) {
    RegisterInputBlobName("in/" + lbn);
    RegisterOutputBlobName("out/" + lbn);
  }
}

std::string CopyOp::ibn2lbn(const std::string& input_blob_name) const {
  return input_blob_name.substr(3);
}
std::string CopyOp::obn2lbn(const std::string& output_blob_name) const {
  return output_blob_name.substr(4);
}

std::string CopyOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  LOG(FATAL) << "This Op doesn't have input_diff_blob_name";
  return "";
}

std::string CopyOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  LOG(FATAL) << "This Op doesn't have output_diff_blob_name";
  return "";
}

} // namespace oneflow
