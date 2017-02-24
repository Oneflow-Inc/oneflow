#ifndef LAYER_MULTINOMIAL_LOGISTIC_LOSS_LAYER_DESC_H_
#define LAYER_MULTINOMIAL_LOGISTIC_LOSS_LAYER_DESC_H_

#include "layer/base_layer_desc.h"

namespace oneflow {

// MLLoss = MultinomialLogisticLoss
class MLLossDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(MLLossDataBlobDescSet);
  MLLossDataBlobDescSet() = default;
  ~MLLossDataBlobDescSet() = default;

  void Init(const std::string& layer_name);

 private:
  BlobDescriptor* data_;
  BlobDescriptor* data_diff_;
  BlobDescriptor* label_;
  BlobDescriptor* label_diff_;
  BlobDescriptor* loss_;
  BlobDescriptor* loss_buffer_;

};

class MLLossModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(MLLossModelBlobDescSet);
  MLLossModelBlobDescSet() = default;
  ~MLLossModelBlobDescSet() = default;
  
  void Init(const std::string& layer_name) {
    ModelBlobDescSet::Init();
  }

 private:

};

class MultinomialLogisticLossLayerDesc : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossLayerDesc);
  MultinomialLogisticLossLayerDesc() = default;
  ~MultinomialLogisticLossLayerDesc() = default;

  void Init(const LayerConf& layer_conf) override;

 private:
  MultinomialLogisticLossLayerConf layer_conf_;

};

} // namespace oneflow

#endif // LAYER_MULTINOMIAL_LOGISTIC_LOSS_LAYER_DESC_H_
