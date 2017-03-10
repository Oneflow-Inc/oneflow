#include "operator/clone_op.h"

namespace oneflow {

namespace {

void InitDataBlobNameSet(
    DataBlobNameSet& cur_set,
    const std::string& logical_blob_name,
    int32_t clone_num) {
  cur_set.input_blob_names.push_back(logical_blob_name);
  cur_set.output_blob_names.assign(clone_num, logical_blob_name);
}

void InitModelBlobNameSet() {
  // do nothing
}

}

std::string CloneOp::ibn2lbn(const std::string& input_blob_name) const {
  return input_blob_name;
}
std::string CloneOp::obn2lbn(const std::string& output_blob_name) const {
  return output_blob_name;
}

std::string CloneOp::idbn2lbn(const std::string input_diff_blob_name) const {
  LOG(FATAL) << "TODO";
  return "";
}
std::string CloneOp::odbn2lbn(const std::string output_diff_blob_name) const {
  LOG(FATAL) << "TODO";
  return "";
}

void CloneOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();

  CHECK(op_conf.has_clone_op_conf());
  auto cnf_ptr = new CloneOpConf(op_conf.clone_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  InitDataBlobNameSet(mutable_data_blob_name_set(),
                      cnf_ptr->logical_blob_name(),
                      cnf_ptr->clone_num());
  InitModelBlobNameSet();
}


} // namespace oneflow
