#ifndef _LAYERS_CONCAT_LAYER_H_
#define _LAYERS_CONCAT_LAYER_H_
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class ConcatData : public DataParam<Dtype> {
public:
  std::vector<Blob<Dtype>*> in;
  std::vector<Blob<Dtype>*> in_diff;
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  explicit ConcatData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
  }
  void SetInputNum(const std::string& layer_name, int32_t in_num) {
    in.resize(in_num, nullptr);
    in_diff.resize(in_num, nullptr);
    for (int32_t idx = 0; idx < in_num; ++idx) {
      DATA_REGISTER_ARRAY_BLOB(layer_name, in, idx, BlobType::kInput);
      DATA_REGISTER_ARRAY_BLOB(layer_name, in_diff, idx, BlobType::kInDiff);
    }
  }
};

template <typename Dtype>
class ConcatModel : public ModelParam<Dtype> {
public:
  explicit ConcatModel(const std::string& layer_name) {}
};

template <typename Dtype>
class ConcatParam : public LayerParam<Dtype> {
public:
  // Init from proto
  int32_t in_num_;
  int32_t axis_;
  // Init from input shape
  int64_t num_concats_;
  int64_t concat_input_size_;

  explicit ConcatParam() {}

  void SetInputNum(const std::string& layer_name, int32_t in_num) {
    in_num_ = in_num;
    CHECK_GT(in_num_, 1);
    GET_CONCRETE_POINTER(ConcatData, prototype_data, prototype_data_);
    prototype_data->SetInputNum(layer_name, in_num_);
  }
};

template <typename Dtype>
class ConcatLayer : public BaseLayer<Dtype> {
public:
  explicit ConcatLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    ConcatData<Dtype>* data = new ConcatData<Dtype>(layer_name_);
    GET_CONCRETE_POINTER(ConcatParam, param, param_);
    CHECK_GT(param->in_num_, 1);
    data->SetInputNum(layer_name_, param->in_num_);
    return data;
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new ConcatModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:

  ConcatLayer(const ConcatLayer& other) = delete;
  ConcatLayer& operator=(const ConcatLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_CONCAT_LAYER_H_