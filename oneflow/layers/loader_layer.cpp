#include <cstdint>
#include <string>
#include <vector>
#include "layers/loader_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "memory/blob.h"
#include "context/one.h"

namespace oneflow {
template <typename Dtype>
void LoaderLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new LoaderParam<Dtype>();

  LoaderProto loader_proto;
  ParseProtoFromStringOrDie(proto_param_, &loader_proto);
  // Get channel_num, height, width
  param->data_path_ = loader_proto.source();
  param->piece_size_ = loader_proto.piece_size();
  param->channel_num_ = loader_proto.channel();
  param->height_ = loader_proto.height();
  param->width_ = loader_proto.width();
  param_ = param;
}
template <typename Dtype>
void LoaderLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(LoaderData, data, data_param);
  GET_CONCRETE_POINTER(LoaderParam, param, param_);

  // Must get piece size through |SetPieceSize| before |InitFromInputShape| 
  CHECK(param->piece_size_ > 0)
    << "Please call |SetPieceSize| firstly to set piece_size.";

  std::vector<int64_t> data_shape{
    param->piece_size_, param->channel_num_, param->height_, param->width_ };
  data->data->set_shape(data_shape);
  std::vector<int64_t> label_shape{ param->piece_size_, 1 };
  data->label->set_shape(label_shape);

  // Align the blob shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}
template <typename Dtype>
void LoaderLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(LoaderData, data, data_param);
  GET_CONCRETE_POINTER(LoaderModel, model, model_param);
  GET_CONCRETE_POINTER(LoaderParam, param, param_);
  // Use ctx, data and model

}
template <typename Dtype>
void LoaderLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(LoaderData, data, data_param);
  GET_CONCRETE_POINTER(LoaderModel, model, model_param);
  // Use ctx, data and model
}
INSTANTIATE_LAYER_FUNCS(LoaderLayer);
INSTANTIATE_CLASS(LoaderLayer);
REGISTER_LAYER_CLASS(Loader);
}
