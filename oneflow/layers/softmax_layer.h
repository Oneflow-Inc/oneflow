#ifndef _LAYERS_SOFTMAX_LAYER_H_
#define _LAYERS_SOFTMAX_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class SoftmaxData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  explicit SoftmaxData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
  }
};
template <typename Dtype>
class SoftmaxModel : public ModelParam<Dtype> {
public:
  Blob<Dtype>* scale{ nullptr };
  explicit SoftmaxModel(const std::string& layer_name) {
    MODEL_REGISTER_BLOB(layer_name, scale, BlobType::kTemp);
  }
};

template <typename Dtype>
class SoftmaxParam : public LayerParam<Dtype> {
public:
  // Init from proto
  int32_t axis_;
  // Init from input shape
  int32_t outer_num_;
  int32_t inner_num_;
  int32_t softmax_axis_;
  explicit SoftmaxParam() {
  }
};

template <typename Dtype>
class SoftmaxLayer : public BaseLayer<Dtype> {
public:
  explicit SoftmaxLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new SoftmaxData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new SoftmaxModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  SoftmaxLayer(const SoftmaxLayer& other) = delete;
  SoftmaxLayer& operator=(const SoftmaxLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_SOFTMAX_LAYER_H_