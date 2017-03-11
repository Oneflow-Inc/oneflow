#include <cstdint>
#include <vector>
#include <string>
#include "layers/model_update_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void ModelUpdateLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new ModelUpdateParam<Dtype>();
  ModelUpdateProto model_update_proto;
  ParseProtoFromStringOrDie(proto_param_, &model_update_proto);
  // TODO(jiyuan): set param properties if necessary
  param_ = param;
}
template <typename Dtype>
void ModelUpdateLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(ModelUpdateData, data, data_param);
  GET_CONCRETE_POINTER(ModelUpdateParam, param, param_);
  // TODO(jiyuan): set blob shape
}
template <typename Dtype>
void ModelUpdateLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ModelUpdateData, data, data_param);
  GET_CONCRETE_POINTER(ModelUpdateModel, model, model_param);
  GET_CONCRETE_POINTER(ModelUpdateParam, param, param_);
}
template <typename Dtype>
void ModelUpdateLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ModelUpdateData, data, data_param);
  GET_CONCRETE_POINTER(ModelUpdateModel, model, model_param);
  GET_CONCRETE_POINTER(ModelUpdateParam, param, param_);
  // Use ctx, data and model
}
INSTANTIATE_LAYER_FUNCS(ModelUpdateLayer);
INSTANTIATE_CLASS(ModelUpdateLayer);
REGISTER_LAYER_CLASS(ModelUpdate);
}  // namespace caffe
