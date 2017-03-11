#include <cstdint>
#include <string>
#include <vector>
#include "layers/softmax_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void SoftmaxLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new SoftmaxParam<Dtype>();
  SoftmaxProto softmax_proto;
  ParseProtoFromStringOrDie(proto_param_, &softmax_proto);
  param->axis_ = softmax_proto.axis();

  param_ = param;
}
template <typename Dtype>
void SoftmaxLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(SoftmaxData, data, data_param);
  GET_CONCRETE_POINTER(SoftmaxParam, param, param_);
  auto model_param = param->mutable_model_param();
  GET_CONCRETE_POINTER(SoftmaxModel, model, model_param);

  const Shape& in_shape = data->in->shape();
  data->in_diff->set_shape(in_shape);
  data->out->set_shape(in_shape);
  data->out_diff->set_shape(in_shape);


  param->softmax_axis_ = in_shape.CanonicalAxisIndex(param->axis_);
  param->outer_num_ = in_shape.count(0, param->softmax_axis_);
  param->inner_num_ = in_shape.count(param->softmax_axis_ + 1);

  std::vector<int64_t> scale_dims = in_shape.shape();
  scale_dims[param->softmax_axis_] = 1;
  model->scale->set_shape(scale_dims);

  // NOTE(jiyuan): remember to align the shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}
INSTANTIATE_CLASS(SoftmaxLayer);
REGISTER_LAYER_CLASS(Softmax);
}  // namespace caffe
