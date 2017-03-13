#ifndef _LAYERS_STORE_LAYER_H_
#define _LAYERS_STORE_LAYER_H_

#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class StoreData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };

  explicit StoreData(const std::string layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
  }
};

template <typename Dtype>
class StoreModel : public ModelParam<Dtype> {
public:
  explicit StoreModel(const std::string layer_name) { }
};

template <typename Dtype>
class StoreParam : public LayerParam<Dtype> {
public:
  bool stop_{ false };
  std::vector<std::string> store_layer_names;
  std::vector<int64_t> store_layer_shapes;
  std::vector<int64_t> layer_seek_pos;
  explicit StoreParam() {}
};

template <typename Dtype>
class StoreLayer : public BaseLayer<Dtype> {
public:
  explicit StoreLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new StoreData<Dtype>(layer_name_);
  }

  ModelParam<Dtype>* CreateModelParam() const override {
    return new StoreModel<Dtype>(layer_name_);
  }

  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  StoreLayer(const StoreLayer& other) = delete;
  StoreLayer&
    operator= (const StoreLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYER_GENERATE_GRADIENT_LAYER_H_
