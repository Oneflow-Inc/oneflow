#include "operator/convolution_op.h"
#include "glog/logging.h"

namespace oneflow {

void ConvolutionDataBlobDescSet::Init(const std::string& op_name) {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr(op_name + "/in", &in_);
  RegisterInputDiffBlobPptr(op_name + "/in_diff", &in_diff_);
  RegisterOutputBlobPptr(op_name + "/out", &out_);
  RegisterOutputDiffBlobPptr(op_name + "/out_diff", &out_diff_);
  RegisterDataTmpBlobPptr(op_name + "/col_buf", &col_buf_);
}

void ConvolutionModelBlobDescSet::Init(const std::string& op_name) {
  ModelBlobDescSet::Init();
  RegisterModelBlobPptr(op_name + "/weight", &weight_);
  RegisterModelDiffBlobPptr(op_name + "/weight_diff", &weight_diff_);
  RegisterModelBlobPptr(op_name + "/bias", &bias_);
  RegisterModelDiffBlobPptr(op_name + "/bias_diff", &bias_diff_);
  RegisterModelTmpBlobPptr(op_name + "/bias_multiplier", &bias_multiplier_);
}

void ConvolutionOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_convolution_op_conf());
  auto cnf_ptr = new ConvolutionOpConf(op_conf.convolution_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  auto data_ptr = new ConvolutionDataBlobDescSet();
  data_ptr->Init(op_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new ConvolutionModelBlobDescSet();
  model_ptr->Init(op_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
