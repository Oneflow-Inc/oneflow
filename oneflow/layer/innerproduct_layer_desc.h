#ifndef LAYER_INNERPRODUCT_LAYER_DESC_H_
#define LAYER_INNERPRODUCT_LAYER_DESC_H_

#include "layer/base_layer_desc.h"

namespace oneflow {

class InnerProductDataBlobDescSet final : public DataBlobDescSet {
 public:
   DISALLOW_COPY_AND_MOVE(InnerProductDataBlobDescSet);
   InnerProductDataBlobDescSet() = default;
   ~InnerProductDataBlobDescSet() = default;

   void Init(const std::string& layer_name);

 private:
   BlobDescriptor* in_;
   BlobDescriptor* in_diff_;
   BlobDescriptor* out_;
   BlobDescriptor* out_diff_;
};

class InnerProductModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(InnerProductModelBlobDescSet);
  InnerProductModelBlobDescSet() = default;
  ~InnerProductModelBlobDescSet() = default;
  
  void Init(const std::string& layer_name);

 private:
  BlobDescriptor* weight_;
  BlobDescriptor* weight_diff_;
  BlobDescriptor* bias_;
  BlobDescriptor* bias_diff_;
  BlobDescriptor* bias_multiplier_;

};

class InnerProductLayerDesc final : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(InnerProductLayerDesc);
  InnerProductLayerDesc() = default;
  ~InnerProductLayerDesc() = default;

  void Init(const LayerConf& layer_conf) override;

 private:
  InnerProductLayerConf layer_conf_;

};

} // namespace oneflow

#endif // LAYER_INNERPRODUCT_LAYER_DESC_H_
