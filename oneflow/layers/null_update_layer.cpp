#include <cstdint>
#include <vector>
#include <string>
#include "layers/null_update_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void NullUpdateLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new NullUpdateParam<Dtype>();
  NullUpdateProto null_update_proto;
  ParseProtoFromStringOrDie(proto_param_, &null_update_proto);
  // TODO(jiyuan): set param properties if necessary
  param_ = param;
}
template <typename Dtype>
void NullUpdateLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(NullUpdateData, data, data_param);
  GET_CONCRETE_POINTER(NullUpdateParam, param, param_);
  // TODO(jiyuan): set the blob shape
}
template <typename Dtype>
void NullUpdateLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(NullUpdateData, data, data_param);
  GET_CONCRETE_POINTER(NullUpdateModel, model, model_param);
  GET_CONCRETE_POINTER(NullUpdateParam, param, param_);
  // Use ctx, data and model
}
template <typename Dtype>
void NullUpdateLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(NullUpdateData, data, data_param);
  GET_CONCRETE_POINTER(NullUpdateModel, model, model_param);
  GET_CONCRETE_POINTER(NullUpdateParam, param, param_);
  // Use ctx, data and model
}
INSTANTIATE_LAYER_FUNCS(NullUpdateLayer);
INSTANTIATE_CLASS(NullUpdateLayer);
REGISTER_LAYER_CLASS(NullUpdate);
}  // namespace caffe
