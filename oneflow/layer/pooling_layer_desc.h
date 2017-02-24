#ifndef LAYER_POOLING_LAYER_DESC_H_
#define LAYER_POOLING_LAYER_DESC_H_

#include "layer/base_layer_desc.h"

namespace oneflow {

class PoolingDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(PoolingDataBlobDescSet);
  PoolingDataBlobDescSet() = default;
  ~PoolingDataBlobDescSet() = default;

  void Init(const std::string& layer_name) {
    DataBlobDescSet::Init();
    RegisterInputBlobPptr(layer_name + "/in", &in_);
    RegisterInputDiffBlobPptr(layer_name + "/in_diff", &in_diff_);
    RegisterOutputBlobPptr(layer_name + "/out", &out_);
    RegisterOutputDiffBlobPptr(layer_name + "/out_diff", &out_diff_);
    RegisterDataTmpBlobPptr(layer_name + "/idx", &idx_);
  }

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;
  BlobDescriptor* idx_;

};

class PoolingModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(PoolingModelBlobDescSet);
  PoolingModelBlobDescSet() = default;
  ~PoolingModelBlobDescSet() = default;

  void Init(const std::string& layer_name) {
    ModelBlobDescSet::Init();
  }

 private:
};

class PoolingLayerDesc final : public BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(PoolingLayerDesc);
  PoolingLayerDesc() = default;
  ~PoolingLayerDesc() = default;

  void Init(const LayerConf& layer_conf) override;

 private:
  PoolingLayerConf layer_conf_;

};

} // namespace oneflow

#endif // LAYER_POOLING_LAYER_DESC_H_
