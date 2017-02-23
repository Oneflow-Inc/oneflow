#ifndef LAYER_LOADER_LAYER_DESC_H_
#define LAYER_LOADER_LAYER_DESC_H_

#include "layer/base_layer_desc.cpp"

namespace oneflow {

class LoaderDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderDataBlobDescSet);
  LoaderDataBlobDescSet() = default;
  ~LoaderDataBlobDescSet() = default;

  void Init(const std::string& layer_name) {
    DataBlobDescSet::Init();
    RegisterOutputBlobPptr(layer_name + ".data", &data_);
    RegisterOutputBlobPptr(layer_name + ".label", &label_);
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

  void Init() {
    ModelBlobDescSet::Init();
  }

 private:
};

class LoaderLayerDesc final : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderLayerDesc);
  LoaderLayerDesc() = default;
  ~LoaderLayerDesc() = default;
  
  void Init(const LayerConf& layer_conf) override {
    BaseLayerDesc::Init();
    mutable_layer_name() = layer_conf.name();
    CHECK(layer_conf.has_loader_layer_conf());
    layer_conf_ = layer_conf.loader_layer_conf();
    
    auto data_ptr = new LoaderDataBlobDescSet();
    data_ptr->Init(layer_name());
    mutable_data_blob_desc_set().reset(data_ptr);

    auto model_ptr = new LoaderModelBlobDescSet();
    model_ptr->Init(layer_name());
    mutable_model_blob_desc_set().reset(model_ptr);
  }

 private:
  LoaderLayerConf layer_conf_;

};

} // namespace oneflow

#endif // LAYER_LOADER_LAYER_DESC_H_
