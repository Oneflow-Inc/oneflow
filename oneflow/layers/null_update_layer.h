#ifndef _LAYERS_NULL_UPDATE_LAYER_H_
#define _LAYERS_NULL_UPDATE_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class NullUpdateData : public DataParam<Dtype> {
public:
  Blob<Dtype>* weight{ nullptr };
  explicit NullUpdateData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, weight, BlobType::kInput);
  }
};

template <typename Dtype>
class NullUpdateModel : public ModelParam<Dtype> {
public:
  explicit NullUpdateModel(const std::string& layer_name) {}
};

template <typename Dtype>
class NullUpdateParam : public LayerParam<Dtype> {
public:
  // Init from proto
  explicit NullUpdateParam() {}
};

template <typename Dtype>
class NullUpdateLayer : public BaseLayer<Dtype> {
public:
  explicit NullUpdateLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new NullUpdateData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new NullUpdateModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  NullUpdateLayer(const NullUpdateLayer& other) = delete;
  NullUpdateLayer& operator=(const NullUpdateLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_NULL_UPDATE_LAYER_H_