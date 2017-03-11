#include <cstdint>
#include <vector>
#include <string>
#include "layers/copy_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void CopyLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new CopyParam<Dtype>();
  CopyProto copy_proto;
  ParseProtoFromStringOrDie(proto_param_, &copy_proto);
  CHECK(copy_proto.has_num());
  param->num_ = copy_proto.num();
  CHECK(copy_proto.has_copy_type());
  param->copy_type_ = copy_proto.copy_type();
  param_ = param;
}
template <typename Dtype>
void CopyLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(CopyData, data, data_param);
  GET_CONCRETE_POINTER(CopyParam, param, param_);
  for (int32_t i = 0; i < param->num_; ++i) {
    data->out[i]->mutable_shape() = data->in[i]->shape();
  }
}
template <typename Dtype>
void CopyLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(CopyData, data, data_param);
  GET_CONCRETE_POINTER(CopyModel, model, model_param);
  GET_CONCRETE_POINTER(CopyParam, param, param_);
  // Use ctx, data and model
  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  cudaMemcpyKind memcpy_kind;
  switch (param->copy_type_) {
  case ForwardH2D:
    memcpy_kind = cudaMemcpyHostToDevice;
    break;
  case ForwardD2H:
    memcpy_kind = cudaMemcpyDeviceToHost;
    break;
  case ForwardD2D:
    memcpy_kind = cudaMemcpyDeviceToDevice;
    break;
  }
  for (int i = 0; i < data->out.size(); ++i) {
    CUDA_CHECK(cudaMemcpyAsync(data->out[i]->mutable_data(),
      data->in[i]->data(), data->in[i]->shape().count() * sizeof(Dtype),
      memcpy_kind, ctx.cuda_stream));
  }
}
template <typename Dtype>
void CopyLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(CopyData, data, data_param);
  GET_CONCRETE_POINTER(CopyModel, model, model_param);
  GET_CONCRETE_POINTER(CopyParam, param, param_);
  // Use ctx, data and model
  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  cudaMemcpyKind memcpy_kind;
  switch (param->copy_type_) {
  case ForwardH2D:
    memcpy_kind = cudaMemcpyDeviceToHost;
    break;
  case ForwardD2H:
    memcpy_kind = cudaMemcpyHostToDevice;
    break;
  case ForwardD2D:
    memcpy_kind = cudaMemcpyDeviceToDevice;
    break;
  }

  for (int i = 0; i < data->out.size(); ++i) {
    if (!data->channel_is_enabled(i)) continue;
    CUDA_CHECK(cudaMemcpyAsync(data->in[i]->mutable_data(),
      data->out[i]->data(), data->out[i]->shape().count() * sizeof(Dtype),
      memcpy_kind, ctx.cuda_stream));
  }
}
INSTANTIATE_LAYER_FUNCS(CopyLayer);
INSTANTIATE_CLASS(CopyLayer);
REGISTER_LAYER_CLASS(Copy);
}  // namespace caffe
