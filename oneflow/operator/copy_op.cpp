#include "operator/copy_op.h"

namespace oneflow {

void CopyDataBlobDescSet::Init(
    const google::protobuf::RepeatedPtrField<std::string>& lbns) {
  DataBlobDescSet::Init();
  input_blobs_.resize(lbns.size());
  output_blobs_.resize(lbns.size());
  for (int i = 0; i < lbns.size(); ++i) {
    RegisterInputBlobPptr("in/" + lbns.Get(i), &(input_blobs_[i]));
    RegisterOutputBlobPptr("out/" + lbns.Get(i), &(output_blobs_[i]));
  }
}

std::string CopyOp::ibn2lbn(const std::string& input_blob_name) const {
  return input_blob_name.substr(3);
}
std::string CopyOp::obn2lbn(const std::string& output_blob_name) const {
  return output_blob_name.substr(4);
}

void CopyOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();

  CHECK(op_conf.has_copy_op_conf());
  auto cnf_ptr = new CopyOpConf(op_conf.copy_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  auto data_ptr = new CopyDataBlobDescSet();
  data_ptr->Init(cnf_ptr->logical_blob_names());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new CopyModelBlobDescSet();
  model_ptr->Init();
  mutable_model_blob_desc_set().reset(model_ptr);
}


} // namespace oneflow
