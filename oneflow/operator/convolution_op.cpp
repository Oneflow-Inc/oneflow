#include "operator/convolution_op.h"
#include "glog/logging.h"

namespace oneflow {

void ConvolutionDataBlobDescSet::Init() {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr("in", &in_);
  RegisterInputDiffBlobPptr("in_diff", &in_diff_);
  RegisterOutputBlobPptr("out", &out_);
  RegisterOutputDiffBlobPptr("out_diff", &out_diff_);
  RegisterDataTmpBlobPptr("col_buf", &col_buf_);
}

void ConvolutionModelBlobDescSet::Init() {
  ModelBlobDescSet::Init();
  RegisterModelBlobPptr("weight", &weight_);
  RegisterModelDiffBlobPptr("weight_diff", &weight_diff_);
  RegisterModelBlobPptr("bias", &bias_);
  RegisterModelDiffBlobPptr("bias_diff", &bias_diff_);
  RegisterModelTmpBlobPptr("bias_multiplier", &bias_multiplier_);
}

void ConvolutionOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_convolution_op_conf());
  auto cnf_ptr = new ConvolutionOpConf(op_conf.convolution_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  auto data_ptr = new ConvolutionDataBlobDescSet();
  data_ptr->Init();
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new ConvolutionModelBlobDescSet();
  model_ptr->Init();
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
