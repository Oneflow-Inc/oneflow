//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_relu_layer.h"
#include "layers/layer_factory.h"
namespace caffe {

template <typename Dtype>
void CuDNNReLULayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new CuDNNReLUParam<Dtype>();
  ReLUProto relu_proto;
  ParseProtoFromStringOrDie(proto_param_, &relu_proto);
  param->negative_slope_ = relu_proto.negative_slope();
  param_ = param;
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  
  GET_CONCRETE_POINTER(ReLUData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNReLUParam, param, param_);

  const Shape& in_shape = data->in->shape();
  data->out->set_shape(in_shape);
  data->out_diff->set_shape(in_shape);
  data->in_diff->set_shape(in_shape);
 
  // initialize cuDNN
  cudnn::createTensor4dDesc<Dtype>(&(param->in_desc_));
  cudnn::createTensor4dDesc<Dtype>(&(param->out_desc_));

  const int N = in_shape.num();
  const int K = in_shape.channels();
  const int H = in_shape.height();
  const int W = in_shape.width();
  cudnn::setTensor4dDesc<Dtype>(&(param->in_desc_), N, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&(param->out_desc_), N, K, H, W);

  // NOTE(jiyuan): remember to align the shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}


INSTANTIATE_CLASS(CuDNNReLULayer);
REGISTER_LAYER_CLASS(CuDNNReLU);

}  // namespace caffe
//#endif

#if 0

template <typename Dtype>
CuDNNReLULayer<Dtype>::~CuDNNReLULayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(this->in_desc_);
  cudnnDestroyTensorDescriptor(this->out_desc_);
  cudnnDestroy(this->handle_);
}
#endif