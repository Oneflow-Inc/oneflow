//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_lrn_layer.h"
#include "layers/layer_factory.h"


namespace caffe {


template <typename Dtype>
void CuDNNLRNLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new CuDNNLRNParam<Dtype>();

  LRNProto lrn_proto;
  ParseProtoFromStringOrDie(proto_param_, &lrn_proto);

  param->size_ = lrn_proto.local_size();
  CHECK_EQ(param->size_ % 2, 1) << "LRN only supports odd values for local_size";
  param->pre_pad_ = (param->size_ - 1) / 2;
  param->alpha_ = lrn_proto.alpha();
  param->beta_ = lrn_proto.beta();
  param->k_ = lrn_proto.k();
  param->norm_region_ = lrn_proto.norm_region();

  param_ = param;
}

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::InitFromInputShape(DataParam<Dtype>* data_param) {

  GET_CONCRETE_POINTER(LRNData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNLRNParam, param, param_);

  const Shape& in_shape = data->in->shape();
  CHECK_EQ(4, in_shape.num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";

  param->num_ = in_shape.num();
  param->channels_ = in_shape.channels();
  param->height_ = in_shape.height();
  param->width_ = in_shape.width();

  data->out->set_shape(in_shape);
  data->scale->set_shape(in_shape);
  data->out_diff->set_shape(in_shape);
  data->in_diff->set_shape(in_shape);

  CUDNN_CHECK(cudnnCreateLRNDescriptor(&(param->norm_desc_)));
  cudnn::createTensor4dDesc<Dtype>(&(param->in_desc_));
  cudnn::createTensor4dDesc<Dtype>(&(param->out_desc_));

  cudnn::setTensor4dDesc<Dtype>(&(param->out_desc_), param->num_,
    param->channels_, param->height_, param->width_);
  cudnn::setTensor4dDesc<Dtype>(&(param->in_desc_), param->num_,
    param->channels_, param->height_, param->width_);
  CUDNN_CHECK(cudnnSetLRNDescriptor(param->norm_desc_, param->size_,
    param->alpha_, param->beta_, param->k_));
}



INSTANTIATE_CLASS(CuDNNLRNLayer);
REGISTER_LAYER_CLASS(CuDNNLRN);



}   // namespace caffe
//#endif
