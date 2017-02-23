#include "layer/loader_layer_desc.h"

namespace oneflow {

void LoaderLayerDesc::Init(const LayerConf& layer_conf) {
  mutable_layer_name() = layer_conf.name();
  CHECK(layer_conf.has_loader_layer_conf());
  layer_conf_ = layer_conf.loader_layer_conf();
  
  auto data_ptr = new LoaderDataBlobDescSet();
  data_ptr->Init(layer_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new LoaderModelBlobDescSet();
  model_ptr->Init(layer_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
