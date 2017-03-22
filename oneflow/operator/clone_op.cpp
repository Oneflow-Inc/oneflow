#include "operator/clone_op.h"

namespace oneflow {

void CloneOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();

  CHECK(op_conf.has_clone_op_conf());
  auto cnf = new CloneOpConf(op_conf.clone_op_conf());
  mutable_pb_op_conf().reset(cnf);
  
  RegisterInputBlobName(cnf->lbn());
  RegisterInputDiffBlobName(GenDiffBlobName(cnf->lbn()));
  for (int32_t i = 0; i < cnf->clone_num(); ++i) {
    std::string obn = cnf->lbn() + "/" + std::to_string(i);
    RegisterOutputBlobName(obn);
    RegisterOutputDiffBlobName(GenDiffBlobName(obn));
  }
}

std::string CloneOp::ibn2lbn(const std::string& input_blob_name) const {
  return input_blob_name;
}
std::string CloneOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  return GenUnDiffBlobName(input_diff_blob_name);
}

std::string CloneOp::obn2lbn(const std::string& output_blob_name) const {
  size_t slash_pos = output_blob_name.rfind('/');
  return output_blob_name.substr(0, slash_pos);
}
std::string CloneOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  return obn2lbn(GenUnDiffBlobName(output_diff_blob_name));
}

} // namespace oneflow
