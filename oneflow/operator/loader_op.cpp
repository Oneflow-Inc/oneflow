#include "operator/loader_op.h"
#include "glog/logging.h"

namespace oneflow {

void LoaderOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_loader_op_conf());
  auto cnf = new LoaderOpConf(op_conf.loader_op_conf());
  mutable_pb_op_conf().reset(cnf);
 
  RegisterOutputBlobName("data");
  RegisterOutputBlobName("label");
}

std::string LoaderOp::ibn2lbn(const std::string& input_blob_name) const {
  LOG(FATAL) << "This Op doesn't have input_blob_name";
  return "";
}

std::string LoaderOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  LOG(FATAL) << "This Op doesn't have input_diff_blob_name";
  return "";
}

} // namespace oneflow
