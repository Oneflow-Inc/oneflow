#ifndef LAYER_SOFTMAX_LAYER_DESC_H_
#define LAYER_SOFTMAX_LAYER_DESC_H_

#include "layer/base_layer_desc.h"

namespace oneflow {

class SoftmaxDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(SoftmaxDataBlobDescSet);
  SoftmaxDataBlobDescSet() = default;
  ~SoftmaxDataBlobDescSet() = default;

  void Init(const std::string& layer_name);

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;

};

class SoftmaxModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(SoftmaxModelBlobDescSet);
  SoftmaxModelBlobDescSet() = default;
  ~SoftmaxModelBlobDescSet() = default;

  void Init(const std::string& layer_name) {
    ModelBlobDescSet::Init();
  }

 private:
};

class SoftmaxLayerDesc : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(SoftmaxLayerDesc);
  SoftmaxLayerDesc() = default;
  ~SoftmaxLayerDesc() = default;

  void Init(const LayerConf& layer_conf) override;

 private:
  SoftmaxLayerConf layer_conf_;

};

} // namespace oneflow

#endif // LAYER_SOFTMAX_LAYER_DESC_H_
