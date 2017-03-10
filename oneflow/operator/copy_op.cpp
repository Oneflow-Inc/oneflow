#include "operator/copy_op.h"

namespace oneflow {

namespace {

void InitDataBlobNameSet(
    DataBlobNameSet& cur_set,
    const google::protobuf::RepeatedPtrField<std::string>& lbns) {
  for (const std::string& lbn : lbns) {
    cur_set.input_blob_names.push_back(lbn);
    cur_set.output_blob_names.push_back(lbn);
  }
}

void InitModelBlobNameSet() {
  // do nothing
}

}

std::string CopyOp::ibn2lbn(const std::string& input_blob_name) const {
  return input_blob_name;
}
std::string CopyOp::obn2lbn(const std::string& output_blob_name) const {
  return output_blob_name;
}

std::string CopyOp::idbn2lbn(const std::string input_diff_blob_name) const {
  LOG(FATAL) << "TODO";
  return "";
}
std::string CopyOp::odbn2lbn(const std::string output_diff_blob_name) const {
  LOG(FATAL) << "TODO";
  return "";
}

void CopyOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();

  CHECK(op_conf.has_copy_op_conf());
  auto cnf_ptr = new CopyOpConf(op_conf.copy_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  InitDataBlobNameSet(mutable_data_blob_name_set(),
                      cnf_ptr->logical_blob_names());
  InitModelBlobNameSet();
}


} // namespace oneflow
