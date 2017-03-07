#include "operator/relu_op.h"
#include "glog/logging.h"

namespace oneflow {

void ReluDataBlobDescSet::Init() {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr("in", &in_);
  RegisterInputDiffBlobPptr("in_diff", &in_diff_);
  RegisterOutputBlobPptr("out", &out_);
  RegisterOutputDiffBlobPptr("out_diff", &out_diff_);
}

void ReluOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_relu_op_conf());
  auto cnf_ptr = new ReluOpConf(op_conf.relu_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  auto data_ptr = new ReluDataBlobDescSet();
  data_ptr->Init();
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new ReluModelBlobDescSet();
  model_ptr->Init();
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
