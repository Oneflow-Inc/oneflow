#include "operator/clone_op.h"

namespace oneflow {

void CloneOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_clone_op_conf());
  auto cnf = new CloneOpConf(op_conf.clone_op_conf());
  mut_pb_op_conf().reset(cnf);
  
  RegisterInputBlobName(cnf->lbn());
  for (int32_t i = 0; i < cnf->clone_num(); ++i) {
    std::string obn = cnf->lbn() + "/" + std::to_string(i);
    RegisterOutputBlobName(obn);
  }
}

std::string CloneOp::ibn2lbn(const std::string& input_blob_name) const {
  return input_blob_name;
}

std::string CloneOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  LOG(FATAL) << "Not Valid";
  return "";
}

std::string CloneOp::obn2lbn(const std::string& output_blob_name) const {
  size_t slash_pos = output_blob_name.rfind('/');
  return output_blob_name.substr(0, slash_pos);
}

std::string CloneOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  LOG(FATAL) << "Not Valid";
  return "";
}

} // namespace oneflow
