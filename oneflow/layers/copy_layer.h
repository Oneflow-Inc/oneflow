#ifndef _LAYERS_COPY_LAYER_H_
#define _LAYERS_COPY_LAYER_H_

#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class CopyData : public DataParam<Dtype> {
public:
  std::vector<Blob<Dtype>*> in;
  std::vector<Blob<Dtype>*> out;
  explicit CopyData(const std::string& layer_name) { }
  void SetInputNum(const std::string& layer_name, int32_t in_num) {
    in.resize(in_num, nullptr);
    out.resize(in_num, nullptr);
    enable_channel.resize(in_num, true);
    for (int32_t idx = 0; idx < in_num; ++idx) {
      DATA_REGISTER_ARRAY_BLOB(layer_name, in, idx, BlobType::kInput);
      DATA_REGISTER_ARRAY_BLOB(layer_name, out, idx, BlobType::kOutput);
    }
  }
};

template <typename Dtype>
class CopyModel : public ModelParam<Dtype> {
public:
  explicit CopyModel(const std::string& layer_name) {}
};

template <typename Dtype>
class CopyParam : public LayerParam<Dtype> {
public:
  // Init from proto
  int32_t num_;  // The number of outputs is equal to the number of inputs
  CopyType copy_type_;  // Whether it is in_copy(H2D) or out_copy(D2H), or D2D
  explicit CopyParam() {}

  void SetInputNum(const std::string& layer_name, int32_t in_num) {
    num_ = in_num;
    CHECK_GT(num_, 1);
    GET_CONCRETE_POINTER(CopyData, prototype_data, prototype_data_);
    prototype_data->SetInputNum(layer_name, num_);
  }
};

template <typename Dtype>
class CopyLayer : public BaseLayer<Dtype> {
public:
  explicit CopyLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    CopyData<Dtype>* data = new CopyData<Dtype>(layer_name_);
    GET_CONCRETE_POINTER(CopyParam, param, param_);
    CHECK_GE(param->num_, 1);
    data->SetInputNum(layer_name_, param->num_);
    return data;
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new CopyModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  CopyLayer(const CopyLayer& other) = delete;
  CopyLayer& operator=(const CopyLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_COPY_LAYER_H_