#include "layer/relu_layer_desc.h"
#include "glog/logging.h"

namespace oneflow {

void ReluDataBlobDescSet::Init(const std::string& layer_name) {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr(layer_name + ".in", &in_);
  RegisterInputDiffBlobPptr(layer_name + ".in_diff", &in_diff_);
  RegisterOutputBlobPptr(layer_name + ".out", &out_);
  RegisterOutputDiffBlobPptr(layer_name + ".out_diff", &out_diff_);
}

void ReluLayerDesc::Init(const LayerConf& layer_conf) {
  mutable_layer_name() = layer_conf.name();
  CHECK(layer_conf.has_relu_layer_conf());
  layer_conf_ = layer_conf.relu_layer_conf();

  auto data_ptr = new ReluDataBlobDescSet();
  data_ptr->Init(layer_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new ReluModelBlobDescSet();
  model_ptr->Init(layer_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
