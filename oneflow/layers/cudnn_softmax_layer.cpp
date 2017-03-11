//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_softmax_layer.h"
#include "layers/layer_factory.h"
namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new CuDNNSoftmaxParam<Dtype>();
  SoftmaxProto softmax_proto;
  ParseProtoFromStringOrDie(proto_param_, &softmax_proto);
  param->axis_ = softmax_proto.axis();

  param_ = param;
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(SoftmaxData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNSoftmaxParam, param, param_);
  auto model_param = param->mutable_model_param();
  GET_CONCRETE_POINTER(CuDNNSoftmaxModel, model, model_param);

  const Shape& in_shape = data->in->shape();
  data->out_diff->set_shape(in_shape);
  data->out->set_shape(in_shape);
  data->in_diff->set_shape(in_shape);


  param->softmax_axis_ = in_shape.CanonicalAxisIndex(param->axis_);
  param->outer_num_ = in_shape.count(0, param->softmax_axis_);
  param->inner_num_ = in_shape.count(param->softmax_axis_ + 1);

  //std::vector<int64_t> scale_dims = in_shape.shape();
  //scale_dims[param->softmax_axis_] = 1;
  //model->scale->set_shape(scale_dims);

  cudnn::createTensor4dDesc<Dtype>(&(param->in_desc_));
  cudnn::createTensor4dDesc<Dtype>(&(param->out_desc_));

  int N = param->outer_num_;
  int K = in_shape.shape(param->softmax_axis_);
  int H = param->inner_num_;
  int W = 1;
  cudnn::setTensor4dDesc<Dtype>(&(param->in_desc_), N, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&(param->out_desc_), N, K, H, W);

  // NOTE(jiyuan): remember to align the shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);

}


INSTANTIATE_CLASS(CuDNNSoftmaxLayer);
REGISTER_LAYER_CLASS(CuDNNSoftmax);

}  // namespace caffe
//#endif