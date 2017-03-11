#include <vector>

#include "layers/lrn_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {

template <typename Dtype>
void LRNLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new LRNParam<Dtype>();

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
void LRNLayer<Dtype>::InitFromInputShape(DataParam<Dtype>* data_param) {

  GET_CONCRETE_POINTER(LRNData, data, data_param);
  GET_CONCRETE_POINTER(LRNParam, param, param_);

  const Shape& in_shape = data->in->shape();
  CHECK_EQ(4, in_shape.num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";

  param->num_ = in_shape.num();
  param->channels_ = in_shape.channels();
  param->height_ = in_shape.height();
  param->width_ = in_shape.width();



  switch (param->norm_region_) {
  case LRNProto_NormRegion_ACROSS_CHANNELS:
    data->out->set_shape(in_shape);
    data->scale->set_shape(in_shape);
    data->out_diff->set_shape(in_shape);
    data->in_diff->set_shape(in_shape);
    break;
  case LRNProto_NormRegion_WITHIN_CHANNEL:
    // To do ...
    break;
  }
}


INSTANTIATE_CLASS(LRNLayer);
REGISTER_LAYER_CLASS(LRN);
}  // namespace caffe

#if 0
template <typename Dtype>
void LRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
size_ = this->layer_param_.lrn_param().local_size();
CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
pre_pad_ = (size_ - 1) / 2;
alpha_ = this->layer_param_.lrn_param().alpha();
beta_ = this->layer_param_.lrn_param().beta();
k_ = this->layer_param_.lrn_param().k();
if (this->layer_param_.lrn_param().norm_region() ==
  LRNParameter_NormRegion_WITHIN_CHANNEL) {
  // Set up split_layer_ to use inputs in the numerator and denominator.
  split_top_vec_.clear();
  split_top_vec_.push_back(&product_input_);
  split_top_vec_.push_back(&square_input_);
  LayerParameter split_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_param));
  split_layer_->SetUp(bottom, split_top_vec_);
  // Set up square_layer_ to square the inputs.
  square_bottom_vec_.clear();
  square_top_vec_.clear();
  square_bottom_vec_.push_back(&square_input_);
  square_top_vec_.push_back(&square_output_);
  LayerParameter square_param;
  square_param.mutable_power_param()->set_power(Dtype(2));
  square_layer_.reset(new PowerLayer<Dtype>(square_param));
  square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
  // Set up pool_layer_ to sum over square neighborhoods of the input.
  pool_top_vec_.clear();
  pool_top_vec_.push_back(&pool_output_);
  LayerParameter pool_param;
  pool_param.mutable_pooling_param()->set_pool(
    PoolingParameter_PoolMethod_AVE);
  pool_param.mutable_pooling_param()->set_pad(pre_pad_);
  pool_param.mutable_pooling_param()->set_kernel_size(size_);
  pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
  pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
  // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
  // the sum of a squared neighborhood (the output of pool_layer_).
  power_top_vec_.clear();
  power_top_vec_.push_back(&power_output_);
  LayerParameter power_param;
  power_param.mutable_power_param()->set_power(-beta_);
  power_param.mutable_power_param()->set_scale(alpha_);
  power_param.mutable_power_param()->set_shift(Dtype(1));
  power_layer_.reset(new PowerLayer<Dtype>(power_param));
  power_layer_->SetUp(pool_top_vec_, power_top_vec_);
  // Set up a product_layer_ to compute outputs by multiplying inputs by the
  // inverse denominator computed by the power layer.
  product_bottom_vec_.clear();
  product_bottom_vec_.push_back(&product_input_);
  product_bottom_vec_.push_back(&power_output_);
  LayerParameter product_param;
  EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
  product_layer_->SetUp(product_bottom_vec_, top);
}
}

template <typename Dtype>
void LRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
num_ = bottom[0]->num();
channels_ = bottom[0]->channels();
height_ = bottom[0]->height();
width_ = bottom[0]->width();
switch (this->layer_param_.lrn_param().norm_region()) {
case LRNParameter_NormRegion_ACROSS_CHANNELS:
  top[0]->Reshape(num_, channels_, height_, width_);
  scale_.Reshape(num_, channels_, height_, width_);
  break;
case LRNParameter_NormRegion_WITHIN_CHANNEL:
  split_layer_->Reshape(bottom, split_top_vec_);
  square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
  pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
  power_layer_->Reshape(pool_top_vec_, power_top_vec_);
  product_layer_->Reshape(product_bottom_vec_, top);
  break;
}
}




template <typename Dtype>
void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
switch (this->layer_param_.lrn_param().norm_region()) {
case LRNParameter_NormRegion_ACROSS_CHANNELS:
  CrossChannelBackward_cpu(top, propagate_down, bottom);
  break;
case LRNParameter_NormRegion_WITHIN_CHANNEL:
  WithinChannelBackward(top, propagate_down, bottom);
  break;
default:
  LOG(FATAL) << "Unknown normalization region.";
}
}




#endif