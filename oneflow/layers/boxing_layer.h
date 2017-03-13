#ifndef _LAYERS_BOXING_LAYER_H_
#define _LAYERS_BOXING_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class BoxingData : public DataParam<Dtype> {
public:
  // TODO(jiyuan): for boxing layer, could we re-use the data blob and diff blob?
  std::vector<Blob<Dtype>*> in;
  Blob<Dtype>* middle{ nullptr };
  std::vector<Blob<Dtype>*> out;
  explicit BoxingData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, middle, BlobType::kOther);
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
class BoxingModel : public ModelParam<Dtype> {
public:
  explicit BoxingModel(const std::string& layer_name) {}
};

template <typename Dtype>
class BoxingParam : public LayerParam<Dtype> {
public:
  // In-direction property, init from proto
  int32_t in_num_;
  BoxingOp in_op_;
  BoxingOp backward_in_op_;
  int32_t in_axis_;

  // Out-direction property, init from proto
  int32_t out_num_;
  BoxingOp out_op_;
  BoxingOp backward_out_op_;
  int32_t out_axis_;

  explicit BoxingParam() {}
};

template <typename Dtype>
class BoxingLayer : public BaseLayer<Dtype> {
public:
  explicit BoxingLayer(
    const std::string& layer_name, const std::string& proto_param)
    : BaseLayer<Dtype>(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    BoxingData<Dtype>* data = new BoxingData<Dtype>(layer_name_);
    GET_CONCRETE_POINTER(BoxingParam, param, param_);
    CHECK_GE(param->in_num_, 1);
    data->SetInputNum(layer_name_, param->in_num_);
    CHECK_GE(param->out_num_, 1);
    data->SetOutputNum(layer_name_, param->out_num_);
    return data;
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new BoxingModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  BoxingLayer(const BoxingLayer& other) = delete;
  BoxingLayer& operator=(const BoxingLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_BOXING_LAYER_H_