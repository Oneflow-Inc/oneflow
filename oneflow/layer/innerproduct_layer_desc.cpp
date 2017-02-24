#include "layer/innerproduct_layer_desc.h"
#include "glog/logging.h"

namespace oneflow {

void InnerProductDataBlobDescSet::Init(const std::string& layer_name) {
  RegisterInputBlobPptr(layer_name + "/in", &in_);
  RegisterInputDiffBlobPptr(layer_name + "/in_diff", &in_diff_);
  RegisterOutputBlobPptr(layer_name + "/out", &out_);
  RegisterOutputDiffBlobPptr(layer_name + "/out_diff", &out_diff_);
}

void InnerProductModelBlobDescSet::Init(const std::string& layer_name) {
  RegisterModelBlobPptr(layer_name + "/weight", &weight_);
  RegisterModelDiffBlobPptr(layer_name + "/weight_diff", &weight_diff_);
  RegisterModelBlobPptr(layer_name + "/bias", &bias_);
  RegisterModelDiffBlobPptr(layer_name + "/bias_diff_", &bias_diff_);
  RegisterModelTmpBlobPptr(layer_name + "/bias_multiplier",
                           &bias_multiplier_);
}

void InnerProductLayerDesc::Init(const LayerConf& layer_conf) {
  mutable_layer_name() = layer_conf.name();
  
  CHECK(layer_conf.has_inner_product_layer_conf());
  auto cnf_ptr =
      new InnerProductLayerConf(layer_conf.inner_product_layer_conf());
  mutable_pb_layer_conf().reset(cnf_ptr);

  auto data_ptr = new InnerProductDataBlobDescSet();
  data_ptr->Init(layer_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new InnerProductModelBlobDescSet();
  model_ptr->Init(layer_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
