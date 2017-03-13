#ifndef _LAYERS_MODEL_UPDATE_LAYER_H_
#define _LAYERS_MODEL_UPDATE_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class ModelUpdateData : public DataParam<Dtype> {
public:
  Blob<Dtype>* gradient{ nullptr };
  Blob<Dtype>* old_weight{ nullptr };
  Blob<Dtype>* weight{ nullptr };
  explicit ModelUpdateData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, gradient, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, old_weight, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, weight, BlobType::kOutput);
  }
};

template <typename Dtype>
class ModelUpdateModel : public ModelParam<Dtype> {
public:
  explicit ModelUpdateModel(const std::string& layer_name) {}
};

template <typename Dtype>
class ModelUpdateParam : public LayerParam<Dtype> {
public:
  // Init from proto
  // TODO(jiyuan): add something like learning rate
  explicit ModelUpdateParam() {}
};

template <typename Dtype>
class ModelUpdateLayer : public BaseLayer<Dtype> {
public:
  explicit ModelUpdateLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    ModelUpdateData<Dtype>* data = new ModelUpdateData<Dtype>(layer_name_);
    return data;
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new ModelUpdateModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  ModelUpdateLayer(const ModelUpdateLayer& other) = delete;
  ModelUpdateLayer& operator=(const ModelUpdateLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_MODEL_UPDATE_LAYER_H_