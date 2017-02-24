#include "layer/pooling_layer_desc.h"
#include "glog/logging.h"

namespace oneflow {

void PoolingLayerDesc::Init(const LayerConf& layer_conf) {
  mutable_layer_name() = layer_conf.name();
  
  CHECK(layer_conf.has_pooling_layer_conf());
  auto cnf_ptr = new PoolingLayerConf(layer_conf.pooling_layer_conf());
  mutable_pb_layer_conf().reset(cnf_ptr);

  auto data_ptr = new PoolingDataBlobDescSet();
  data_ptr->Init(layer_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new PoolingModelBlobDescSet();
  model_ptr->Init(layer_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
