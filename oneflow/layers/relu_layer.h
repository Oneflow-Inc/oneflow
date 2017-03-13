#ifndef _LAYERS_RELU_LAYER_H_
#define _LAYERS_RELU_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class ReLUData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };

  explicit ReLUData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
  }
};
template <typename Dtype>
class ReLUModel : public ModelParam<Dtype> {
public:
  explicit ReLUModel(const std::string& layer_name) {}
};
template <typename Dtype>
class ReLUParam : public LayerParam<Dtype> {
public:
  Dtype negative_slope_;
  explicit ReLUParam(){
  }
};
template <typename Dtype>
class ReLULayer : public BaseLayer<Dtype> {
public:
  explicit ReLULayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {
    is_elem_wise_ = true;
  }

  DataParam<Dtype>* CreateDataParam() const override {
    return new ReLUData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new ReLUModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  ReLULayer(const ReLULayer& other) = delete;
  ReLULayer& operator=(const ReLULayer& other) = delete;
};
}

#endif  // _LAYERS_RELU_LAYER_H_