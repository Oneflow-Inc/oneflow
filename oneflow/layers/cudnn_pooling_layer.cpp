//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_pooling_layer.h"
#include "layers/layer_factory.h"


namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new CuDNNPoolingParam<Dtype>();

  PoolingProto pooling_proto;
  ParseProtoFromStringOrDie(proto_param_, &pooling_proto);

  if (pooling_proto.global_pooling()) {
    CHECK(!(pooling_proto.has_kernel_size() ||
      pooling_proto.has_kernel_h() || pooling_proto.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  }
  else {
    CHECK(!pooling_proto.has_kernel_size() !=
      !(pooling_proto.has_kernel_h() && pooling_proto.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pooling_proto.has_kernel_size() ||
      (pooling_proto.has_kernel_h() && pooling_proto.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pooling_proto.has_pad() && pooling_proto.has_pad_h()
    && pooling_proto.has_pad_w())
    || (!pooling_proto.has_pad_h() && !pooling_proto.has_pad_w()))
    << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pooling_proto.has_stride() && pooling_proto.has_stride_h()
    && pooling_proto.has_stride_w())
    || (!pooling_proto.has_stride_h() && !pooling_proto.has_stride_w()))
    << "Stride is stride OR stride_h and stride_w are required.";

  param->pool_ = pooling_proto.pool();
  param->global_pooling_ = pooling_proto.global_pooling();
  if (!pooling_proto.has_pad_h()) {
    param->pad_h_ = param->pad_w_ = pooling_proto.pad();
  }
  else {
    param->pad_h_ = pooling_proto.pad_h();
    param->pad_w_ = pooling_proto.pad_w();
  }
  if (!pooling_proto.has_stride_h()) {
    param->stride_h_ = param->stride_w_ = pooling_proto.stride();
  }
  else {
    param->stride_h_ = pooling_proto.stride_h();
    param->stride_w_ = pooling_proto.stride_w();
  }
  if (pooling_proto.has_kernel_size()) {
    param->kernel_h_ = param->kernel_w_ = pooling_proto.kernel_size();
  }
  else {
    param->kernel_h_ = pooling_proto.kernel_h();
    param->kernel_w_ = pooling_proto.kernel_w();
  }

  if (param->pad_h_ != 0 || param->pad_w_ != 0) {
    CHECK(param->pool_
      == PoolingProto_PoolMethod_AVE
      || param->pool_
      == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";
  }
  param_ = param;
}


template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {

  GET_CONCRETE_POINTER(CuDNNPoolingData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNPoolingParam, param, param_);

  const Shape& in_shape = data->in->shape();
  data->in_diff->set_shape(in_shape);
  CHECK_EQ(4, in_shape.num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";
  param->channels_ = in_shape.channels();
  param->height_ = in_shape.height();
  param->width_ = in_shape.width();

  if (param->global_pooling_) {
    CHECK(param->pad_h_ == 0 && param->pad_w_ == 0 && param->stride_h_ == 1 &&
      param->stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
    param->kernel_h_ = in_shape.height();
    param->kernel_w_ = in_shape.width();
  }

  CHECK_GT(param->kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(param->kernel_w_, 0) << "Filter dimensions cannot be zero.";
  CHECK_LT(param->pad_h_, param->kernel_h_);
  CHECK_LT(param->pad_w_, param->kernel_w_);

  param->pooled_height_ = static_cast<int>(ceil(static_cast<float>(
    param->height_ + 2 * param->pad_h_ - param->kernel_h_) /
    param->stride_h_)) + 1;
  param->pooled_width_ = static_cast<int>(ceil(static_cast<float>(
    param->width_ + 2 * param->pad_w_ - param->kernel_w_) /
    param->stride_w_)) + 1;
  if (param->pad_h_ || param->pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((param->pooled_height_ - 1) * param->stride_h_ >=
      param->height_ + param->pad_h_) {
      --param->pooled_height_;
    }
    if ((param->pooled_width_ - 1) * param->stride_w_ >=
      param->width_ + param->pad_w_) {
      --param->pooled_width_;
    }
    CHECK_LT((param->pooled_height_ - 1) * param->stride_h_,
      param->height_ + param->pad_h_);
    CHECK_LT((param->pooled_width_ - 1) * param->stride_w_,
      param->width_ + param->pad_w_);
  }

  // Shape the outputs
  Shape& out_shape = data->out->mutable_shape();
  out_shape.Reshape(
    in_shape.num(), param->channels_, param->pooled_height_, param->pooled_width_);
  Shape& out_diff_shape = data->out_diff->mutable_shape();
  out_diff_shape.Reshape(
    in_shape.num(), param->channels_, param->pooled_height_, param->pooled_width_);

  // Check the pool
  CHECK(param->pool_ == PoolingProto_PoolMethod_AVE
    || param->pool_ == PoolingProto_PoolMethod_MAX);

  // Shape the idx
  // std::vector<int64_t> idx_shape;
  // idx_shape.assign(
  // { in_shape.num(), param->channels_, param->pooled_height_,
  // param->pooled_width_ });
  // data->idx->set_shape(idx_shape);

  cudnn::createTensor4dDesc<Dtype>(&(param->in_desc_));
  cudnn::createTensor4dDesc<Dtype>(&(param->out_desc_));
  cudnn::createPoolingDesc<Dtype>(&(param->pooling_desc_),
    param->pool_, &(param->mode_),
    param->kernel_h_, param->kernel_w_, param->pad_h_, param->pad_w_,
    param->stride_h_, param->stride_w_);

  cudnn::setTensor4dDesc<Dtype>(&(param->in_desc_), in_shape.num(),
    param->channels_, param->height_, param->width_);
  cudnn::setTensor4dDesc<Dtype>(&(param->out_desc_), in_shape.num(),
    param->channels_, param->pooled_height_, param->pooled_width_);

  // NOTE(jiyuan): remember to align the shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);

}




INSTANTIATE_CLASS(CuDNNPoolingLayer);
REGISTER_LAYER_CLASS(CuDNNPooling);



}   // namespace caffe
//#endif
#if 0

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
// Check that handles have been setup before destroying.
if (!handles_setup_) { return; }

cudnnDestroyTensorDescriptor(in_desc_);
cudnnDestroyTensorDescriptor(out_desc_);
cudnnDestroyPoolingDescriptor(pooling_desc_);
cudnnDestroy(handle_);
}
#endif