#ifndef LAYER_RELU_LAYER_DESC_H_
#define LAYER_RELU_LAYER_DESC_H_

#include "layer/base_layer_desc.cpp"

namespace oneflow {

class ReluDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ReluDataBlobDescSet);
  ReluDataBlobDescSet() = default;
  ~ReluDataBlobDescSet() = default;

  void Init(const std::string& layer_name);

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;

};

class ReluModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ReluModelBlobDescSet);
  ReluModelBlobDescSet() = default;
  ~ReluModelBlobDescSet() = default;

  void Init(const std::string& layer_name) {
    ModelBlobDescSet::Init();
  }

 private:
};

class ReluLayerDesc : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(ReluLayerDesc);
  ReluLayerDesc() = default;
  ~ReluLayerDesc() = default;

  void Init(const LayerConf& layer_conf) override;

 private:
  ReluLayerConf layer_conf_;

};

} // namespace oneflow

#endif // LAYER_RELU_LAYER_DESC_H_
