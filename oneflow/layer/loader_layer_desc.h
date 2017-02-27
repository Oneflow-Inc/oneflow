#ifndef LAYER_LOADER_LAYER_DESC_H_
#define LAYER_LOADER_LAYER_DESC_H_

#include "layer/base_layer_desc.h"

namespace oneflow {

class LoaderDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderDataBlobDescSet);
  LoaderDataBlobDescSet() = default;
  ~LoaderDataBlobDescSet() = default;

  void Init(const std::string& layer_name) {
    DataBlobDescSet::Init();
    RegisterOutputBlobPptr(layer_name + "/data", &data_);
    RegisterOutputBlobPptr(layer_name + "/label", &label_);
  }

 private:
  BlobDescriptor* data_;
  BlobDescriptor* label_;

};

class LoaderModelBlobDescSet : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderModelBlobDescSet);
  LoaderModelBlobDescSet() = default;
  ~LoaderModelBlobDescSet() = default;

  void Init(const std::string& layer_name) {
    ModelBlobDescSet::Init();
  }

 private:
};

class LoaderLayerDesc final : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderLayerDesc);
  LoaderLayerDesc() = default;
  ~LoaderLayerDesc() = default;
  
  void Init(const LayerConf& layer_conf) override;
  bool IsElemWise() const override { return false; }

 private:
  LoaderLayerConf layer_conf_;

};

} // namespace oneflow

#endif // LAYER_LOADER_LAYER_DESC_H_
