#ifndef _LAYERS_MULTINOMIAL_LOGISTIC_LOSS_LAYER_H_
#define _LAYERS_MULTINOMIAL_LOGISTIC_LOSS_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template <typename Dtype>
class MultinomialLogisticLossData : public DataParam<Dtype> {
public:
  Blob<Dtype>* data{ nullptr };
  Blob<Dtype>* data_diff{ nullptr };
  Blob<Dtype>* label{ nullptr };
  // NOTE(jiyuan): the |label_diff| actually unnecessary for training the neural
  // network, here we deliberately add it to avoid the complexity of DAG
  // compiling. Otherwise, we must carefully distinguish the feature data and
  // label-data while rewriting the DAGs.
  Blob<Dtype>* label_diff{ nullptr };
  Blob<Dtype>* loss{ nullptr };
  Blob<Dtype>* loss_buffer{ nullptr };
  explicit MultinomialLogisticLossData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, data, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, data_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, label, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, label_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, loss, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, loss_buffer, BlobType::kOther);
  }
};

template <typename Dtype>
class MultinomialLogisticLossModel : public ModelParam<Dtype> {
public:
  Blob<Dtype>* loss_multiplier{ nullptr };
  explicit MultinomialLogisticLossModel(const std::string& layer_name) {
    MODEL_REGISTER_BLOB(layer_name, loss_multiplier, BlobType::kTemp);
  }
};

template <typename Dtype>
class MultinomialLogisticLossParam : public LayerParam<Dtype> {
public:
  explicit MultinomialLogisticLossParam() {
  }
};
template <typename Dtype>
class MultinomialLogisticLossLayer : public BaseLayer<Dtype> {
public:
  explicit MultinomialLogisticLossLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new MultinomialLogisticLossData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new MultinomialLogisticLossModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  MultinomialLogisticLossLayer(
    const MultinomialLogisticLossLayer& other) = delete;
  MultinomialLogisticLossLayer& operator=(
    const MultinomialLogisticLossLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_MULTINOMIAL_LOGISTIC_LOSS_LAYER_H_