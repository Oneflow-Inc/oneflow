#include <cstdint>
#include <vector>
#include "layers/relu_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void ReLULayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new ReLUParam<Dtype>();
  ReLUProto relu_proto;
  ParseProtoFromStringOrDie(proto_param_, &relu_proto);
  param->negative_slope_ = relu_proto.negative_slope();
  param_ = param;
}
template <typename Dtype>
void ReLULayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(ReLUData, data, data_param);
  GET_CONCRETE_POINTER(ReLUParam, param, param_);

  const Shape& in_shape = data->in->shape();
  data->out->set_shape(in_shape);
  data->out_diff->set_shape(in_shape);
  data->in_diff->set_shape(in_shape);

  // NOTE(jiyuan): remember to align the shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}
INSTANTIATE_CLASS(ReLULayer);
REGISTER_LAYER_CLASS(ReLU);
#if 0
template <typename Dtype>
void ReLULayer<Dtype>::LayerSetup(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  DLOG(INFO) << "Setting up ReLULayer...";
}

template <typename Dtype>
void ReLULayer<Dtype>::Reshape(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  CHECK_EQ(1, inputs.size()) << "Currently only support one input";
  CHECK_EQ(1, outputs->size()) << "Currently only support one output";
  const Shape& in_shape = inputs[0]->shape();
  Shape& out_shape = (*outputs)[0]->mutable_shape();
  out_shape.Reshape(in_shape.shape());
}
#endif
}  // namespace caffe
