#ifndef _LAYERS_NET_LAYER_H_
#define _LAYERS_NET_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class NetData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in_envelope{ nullptr };
  std::vector<Blob<Dtype>*> in;
  Blob<Dtype>* out_envelope{ nullptr };
  std::vector<Blob<Dtype>*> out;
  explicit NetData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in_envelope, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out_envelope, BlobType::kOutput);
  }
  void SetInputNum(const std::string& layer_name, int32_t in_num) {
    in.resize(in_num, nullptr);
    for (int32_t idx = 0; idx < in_num; ++idx) {
      DATA_REGISTER_ARRAY_BLOB(layer_name, in, idx, BlobType::kInput);
    }
  }
  void SetOutputNum(const std::string& layer_name, int32_t out_num) {
    out.resize(out_num, nullptr);
    for (int32_t idx = 0; idx < out_num; ++idx) {
      DATA_REGISTER_ARRAY_BLOB(layer_name, out, idx, BlobType::kOutput);
    }
  }
};

template <typename Dtype>
class NetModel : public ModelParam<Dtype> {
public:
  explicit NetModel(const std::string& layer_name) {}
};

template <typename Dtype>
class NetParam : public LayerParam<Dtype> {
public:
  // Init from proto
  int32_t in_num_;
  int32_t out_num_;

  // Whether it is a net node for sending data. true for two cases:
  // (1) out_net in the forward-pass
  // (2) in_net in the backward-pass
  bool forward_is_sender_{ true };

  explicit NetParam() {}
};

template <typename Dtype>
class NetLayer : public BaseLayer<Dtype> {
public:
  explicit NetLayer(
    const std::string& layer_name, const std::string& proto_param)
    : BaseLayer<Dtype>(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    NetData<Dtype>* data = new NetData<Dtype>(layer_name_);
    GET_CONCRETE_POINTER(NetParam, param, param_);
    CHECK_GE(param->in_num_, 1);
    data->SetInputNum(layer_name_, param->in_num_);
    CHECK_GE(param->out_num_, 1);
    data->SetOutputNum(layer_name_, param->out_num_);
    return data;
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new NetModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  NetLayer(const NetLayer& other) = delete;
  NetLayer& operator=(const NetLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_NET_LAYER_H_