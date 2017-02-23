#ifndef LAYER_CONVOLUTION_LAYER_DESC_H_
#define LAYER_CONVOLUTION_LAYER_DESC_H_

#include "layer/base_layer_desc.h"

namespace oneflow {

class ConvolutionDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ConvolutionDataBlobDescSet);
  ConvolutionDataBlobDescSet() = default;
  ~ConvolutionDataBlobDescSet() = default;

  void Init(const std::string& layer_name);

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;
  BlobDescriptor* col_buf_;

};

class ConvolutionModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ConvolutionModelBlobDescSet);
  ConvolutionModelBlobDescSet() = default;
  ~ConvolutionModelBlobDescSet() = default;

  void Init(const std::string& layer_name);

 private:
  BlobDescriptor* weight_;
  BlobDescriptor* weight_diff_;
  BlobDescriptor* bias_;
  BlobDescriptor* bias_diff_;
  BlobDescriptor* bias_multiplier_;
};

class ConvolutionLayerDesc final : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(ConvolutionLayerDesc);
  ConvolutionLayerDesc() = default;
  ~ConvolutionLayerDesc() = default;

  void Init(const LayerConf& layer_conf) override;
  
  const DataBlobDescSet* data_blob_desc_set() override {
    return &data_blob_desc_set_;
  }

  const ModelBlobDescSet* model_blob_desc_set() override {
    return &model_blob_desc_set_;
  }

 private:
  ConvolutionLayerConf layer_conf_;
  ConvolutionDataBlobDescSet data_blob_desc_set_;
  ConvolutionModelBlobDescSet model_blob_desc_set_;

};

} // namespace oneflow

#endif // LAYER_CONVOLUTION_LAYER_DESC_H_
