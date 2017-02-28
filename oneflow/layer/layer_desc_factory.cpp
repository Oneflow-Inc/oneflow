#include "layer/layer_desc_factory.h"
#include "glog/logging.h"
#include "layer/convolution_layer_desc.h"
#include "layer/innerproduct_layer_desc.h"
#include "layer/loader_layer_desc.h"
#include "layer/multinomial_logistic_loss_layer_desc.h"
#include "layer/relu_layer_desc.h"
#include "layer/softmax_layer_desc.h"
#include "layer/pooling_layer_desc.h"

namespace oneflow {

// It is ugly now, maybe we can find one more elegant implemention ?
std::unique_ptr<BaseLayerDesc> LayerDescFactory::ConstructLayerDesc(
    const LayerConf& layer_conf) const {
  std::unique_ptr<BaseLayerDesc> ret;
  switch (layer_conf.specified_type_case()) {
    case LayerConf::kConvolutionLayerConf: {
      ret.reset(new ConvolutionLayerDesc);
    }
    case LayerConf::kInnerProductLayerConf: {
      ret.reset(new InnerProductLayerDesc);
    }
    case LayerConf::kLoaderLayerConf: {
      ret.reset(new LoaderLayerDesc);
    }
    case LayerConf::kPoolingLayerConf: {
      ret.reset(new PoolingLayerDesc);
    }
    case LayerConf::kReluLayerConf: {
      ret.reset(new ReluLayerDesc);
    }
    case LayerConf::kSoftmaxLayerConf: {
      ret.reset(new SoftmaxLayerDesc);
    }
    case LayerConf::kMultinomialLogisticLossLayerConf: {
      ret.reset(new MultinomialLogisticLossLayerDesc);
    }
    default: {
      LOG(FATAL) << "unknow layer";
    }
  }
  ret->Init(layer_conf);
  return ret;
}

} // namespace oneflow
