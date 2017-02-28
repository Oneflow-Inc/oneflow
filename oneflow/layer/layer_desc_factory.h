#ifndef ONEFLOW_LAYER_LAYER_DESC_FACTORY_H_
#define ONEFLOW_LAYER_LAYER_DESC_FACTORY_H_

#include "layer/base_layer_desc.h"
#include "layer/layer_conf.pb.h"

namespace oneflow {

class LayerDescFactory {
 public:
  DISALLOW_COPY_AND_MOVE(LayerDescFactory);
  ~LayerDescFactory() = default;
  static const LayerDescFactory& singleton() {
    static LayerDescFactory obj;
    return obj;
  }
  
  std::unique_ptr<BaseLayerDesc> ConstructLayerDesc(const LayerConf&) const;

 private:
  LayerDescFactory() = default;

};

} // namespace oneflow

#endif // ONEFLOW_LAYER_LAYER_DESC_FACTORY_H_
