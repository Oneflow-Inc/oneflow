#ifndef _LAYERS_LOADER_LAYER_H_
#define _LAYERS_LOADER_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "oneflow.pb.h"
#include "proto_io.h"

namespace oneflow {
template <typename Dtype>
class LoaderData : public DataParam<Dtype> {
public:
  Blob<Dtype>* data{ nullptr };
  Blob<Dtype>* label{ nullptr };
  explicit LoaderData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, data, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, label, BlobType::kOutput);
  }
};
template <typename Dtype>
class LoaderModel : public ModelParam<Dtype> {
public:
  explicit LoaderModel(const std::string& layer_name) {}
};
template <typename Dtype>
class LoaderParam : public LayerParam<Dtype> {
public:
  int32_t piece_size_{ 0 };
  int32_t channel_num_{ 0 };
  int32_t height_{ 0 };
  int32_t width_{ 0 };

  std::string data_path_;

  explicit LoaderParam(){}
};

template <typename Dtype>
class LoaderLayer : public BaseLayer<Dtype> {
public:
  explicit LoaderLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  void SetPieceSize(int32_t piece_size) {
    GET_CONCRETE_POINTER(LoaderParam, param, param_);
    param->piece_size_ = piece_size;
  }
  DataParam<Dtype>* CreateDataParam() const override {
    return new LoaderData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new LoaderModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  LoaderLayer(const LoaderLayer& other) = delete;
  LoaderLayer& operator=(const LoaderLayer& other) = delete;
};
}  // namespace oneflow
#endif  // _LAYERS_LOADER_LAYER_H_