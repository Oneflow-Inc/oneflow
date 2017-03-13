#ifndef _LAYERS_INNERPRODUCT_LAYER_H_
#define _LAYERS_INNERPRODUCT_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class InnerProductData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  explicit InnerProductData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
  }
};

template <typename Dtype>
class InnerProductModel : public ModelParam<Dtype> {
public:
  Blob<Dtype>* weight{ nullptr };
  Blob<Dtype>* weight_diff{ nullptr };
  Blob<Dtype>* bias{ nullptr };
  Blob<Dtype>* bias_diff{ nullptr };
  Blob<Dtype>* bias_multiplier{ nullptr };

  explicit InnerProductModel(const std::string& layer_name) {
    MODEL_REGISTER_BLOB(layer_name, weight, BlobType::kModel);
    MODEL_REGISTER_BLOB(layer_name, weight_diff, BlobType::kModelDiff);
    MODEL_REGISTER_BLOB(layer_name, bias, BlobType::kModel);
    MODEL_REGISTER_BLOB(layer_name, bias_diff, BlobType::kModelDiff);
    MODEL_REGISTER_BLOB(layer_name, bias_multiplier, BlobType::kTemp);
  }
};

template <typename Dtype>
class InnerProductParam : public LayerParam<Dtype> {
public:
  // Init from proto
  int32_t num_output_;  // N_
  bool bias_term_;
  int32_t axis_;

  // Init from input shape
  int32_t num_example_;  // M_
  int32_t num_input_;    // K_

  explicit InnerProductParam() {}
};

template <typename Dtype>
class InnerProductLayer : public BaseLayer<Dtype> {
public:
  explicit InnerProductLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new InnerProductData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new InnerProductModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  InnerProductLayer(const InnerProductLayer& other) = delete;
  InnerProductLayer& operator=(const InnerProductLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_INNERPRODUCT_LAYER_H_