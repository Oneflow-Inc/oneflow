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
std::shared_ptr<BaseLayerDesc> LayerDescFactory::ConstructLayerDesc(
    const LayerConf& layer_conf) const {
  std::shared_ptr<BaseLayerDesc> ret;
  switch (layer_conf.specified_type_case()) {
    case LayerConf::kConvolutionLayerConf: {
      ret = std::make_shared<ConvolutionLayerDesc> ();
    }
    case LayerConf::kInnerProductLayerConf: {
      ret = std::make_shared<InnerProductLayerDesc> ();
    }
    case LayerConf::kLoaderLayerConf: {
      ret = std::make_shared<LoaderLayerDesc> ();
    }
    case LayerConf::kPoolingLayerConf: {
      ret = std::make_shared<PoolingLayerDesc> ();
    }
    case LayerConf::kReluLayerConf: {
      ret = std::make_shared<ReluLayerDesc> ();
    }
    case LayerConf::kSoftmaxLayerConf: {
      ret = std::make_shared<SoftmaxLayerDesc> ();
    }
    case LayerConf::kMultinomialLogisticLossLayerConf: {
      ret = std::make_shared<MultinomialLogisticLossLayerDesc> ();
    }
    default: {
      LOG(FATAL) << "unknow layer";
    }
  }
  ret->Init(layer_conf);
  return ret;
}

} // namespace oneflow
