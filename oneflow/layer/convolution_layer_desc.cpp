#include "layer/convolution_layer_desc.h"
#include "glog/logging.h"

namespace oneflow {

void ConvolutionDataBlobDescSet::Init(const std::string& layer_name) {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr(layer_name + ".in", &in_);
  RegisterInputDiffBlobPptr(layer_name + ".in_diff", &in_diff_);
  RegisterOutputBlobPptr(layer_name + ".out", &out_);
  RegisterOutputDiffBlobPptr(layer_name + ".out_diff", &out_diff_);
  RegisterDataTmpBlobPptr(layer_name + ".col_buf", &col_buf_);
}

void ConvolutionModelBlobDescSet::Init(const std::string& layer_name) {
  ModelBlobDescSet::Init();
  RegisterModelBlobPptr(layer_name + ".weight", &weight_);
  RegisterModelDiffBlobPptr(layer_name + ".weight_diff", &weight_diff_);
  RegisterModelBlobPptr(layer_name + ".bias", &bias_);
  RegisterModelDiffBlobPptr(layer_name + ".bias_diff", &bias_diff_);
  RegisterModelTmpBlobPptr(layer_name + ".bias_multiplier", &bias_multiplier_);
}

void ConvolutionLayerDesc::Init(const LayerConf& layer_conf) {
  mutable_layer_name() = layer_conf.name();
  CHECK(layer_conf.has_convolution_layer_conf());
  layer_conf_ = layer_conf.convolution_layer_conf();
  
  auto data_ptr = new ConvolutionDataBlobDescSet();
  data_ptr->Init(layer_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new ConvolutionModelBlobDescSet();
  model_ptr->Init(layer_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
