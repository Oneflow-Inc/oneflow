#include "operator/softmax_op.h"
#include "glog/logging.h"

namespace oneflow {

void SoftmaxDataBlobDescSet::Init() {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr("in", &in_);
  RegisterInputDiffBlobPptr("in_diff", &in_diff_);
  RegisterOutputBlobPptr("out", &out_);
  RegisterOutputDiffBlobPptr("out_diff", &out_diff_);
}

void SoftmaxOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_softmax_op_conf());
  auto cnf_ptr = new SoftmaxOpConf(op_conf.softmax_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  auto data_ptr = new SoftmaxDataBlobDescSet();
  data_ptr->Init();
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new SoftmaxModelBlobDescSet();
  model_ptr->Init();
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
