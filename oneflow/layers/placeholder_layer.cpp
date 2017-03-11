#include <cstdint>
#include <vector>
#include <string>
#include "layers/placeholder_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void PlaceholderLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new PlaceholderParam<Dtype>();

  PlaceholderProto placeholder_proto;
  ParseProtoFromStringOrDie(proto_param_, &placeholder_proto);

  param_ = param;
}
template <typename Dtype>
void PlaceholderLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(PlaceholderData, data, data_param);
  GET_CONCRETE_POINTER(PlaceholderParam, param, param_);
  auto model_param = param->mutable_model_param();
  LOG(FATAL) << "Not allowed to be here";
}
template <typename Dtype>
void PlaceholderLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(PlaceholderData, data, data_param);
  GET_CONCRETE_POINTER(PlaceholderModel, model, model_param);
  LOG(FATAL) << "Not allowed to be here";
}
template <typename Dtype>
void PlaceholderLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(PlaceholderData, data, data_param);
  GET_CONCRETE_POINTER(PlaceholderModel, model, model_param);
  LOG(FATAL) << "Not allowed to be here";
}
INSTANTIATE_LAYER_FUNCS(PlaceholderLayer);
INSTANTIATE_CLASS(PlaceholderLayer);
REGISTER_LAYER_CLASS(Placeholder);
}  // namespace caffe
