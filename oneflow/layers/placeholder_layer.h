#ifndef _LAYER_PLACEHOLDER_LAYER_H_
#define _LAYER_PLACEHOLDER_LAYER_H_

#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class PlaceholderData : public DataParam<Dtype> {
 public:
  Blob<Dtype>* in {nullptr};
  Blob<Dtype>* out {nullptr};

  explicit PlaceholderData(const std::string layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
  }
};

template <typename Dtype>
class PlaceholderModel : public ModelParam<Dtype> {
 public:
  explicit PlaceholderModel(const std::string layer_name) {
  }
};

template <typename Dtype>
class PlaceholderParam : public LayerParam<Dtype> {
 public:
  PlaceholderParam() {}
};

template <typename Dtype>
class PlaceholderLayer : public BaseLayer<Dtype> {
 public:
  explicit PlaceholderLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new PlaceholderData<Dtype>(layer_name_);
  }

  ModelParam<Dtype>* CreateModelParam() const override {
    return new PlaceholderModel<Dtype>(layer_name_);
  }

  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
 private:
  PlaceholderLayer(const PlaceholderLayer& other) = delete;
  PlaceholderLayer& operator= (const PlaceholderLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYER_PLACEHOLDER_LAYER_H_
