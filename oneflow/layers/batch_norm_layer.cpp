#include <vector>

#include "layers/batch_norm_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "context/one.h"
#include "context/config_parser.h"
#include "context/net_descriptor.h"

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new BatchNormParam<Dtype>();

  BatchNormProto batchnorm_proto;
  ParseProtoFromStringOrDie(proto_param_, &batchnorm_proto);

  param->moving_average_fraction_ = batchnorm_proto.moving_average_fraction();

  param->use_global_stats_ =
    TheOne<Dtype>::config_parser()->net_descriptor()->is_train() == false;

  if (batchnorm_proto.has_use_global_stats())
    param->use_global_stats_ = batchnorm_proto.use_global_stats();

  param->eps_ = batchnorm_proto.eps();

  param_ = param;
}

template <typename Dtype>
void BatchNormLayer<Dtype>::InitFromInputShape(DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(BatchNormData, data, data_param);
  GET_CONCRETE_POINTER(BatchNormParam, param, param_);
  auto model_param = param->mutable_model_param();
  GET_CONCRETE_POINTER(BatchNormModel, model, model_param);

  const Shape& in_shape = data->in->shape();
  data->in_diff->set_shape(in_shape);
  CHECK_EQ(4, in_shape.num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";

  if (in_shape.num_axes() == 1)
    param->channels_ = 1;
  else
    param->channels_ = in_shape.shape(1);

  std::vector<int64_t> blob_shape{
    param->channels_,
    1,
    1,
    1 };
  // TODO(depeng): model blobs_ need to be initialized to 0
  model->blobs_[0]->set_shape(blob_shape);
  model->blobs_[1]->set_shape(blob_shape);
  blob_shape[0] = 1;
  model->blobs_[2]->set_shape(blob_shape);

  data->out->set_shape(in_shape);
  data->out_diff->set_shape(in_shape);
  std::vector<int64_t> shape{
    param->channels_,
    1,
    1,
    1 };
  data->mean_->set_shape(shape);
  data->variance_->set_shape(shape);

  data->temp_->set_shape(in_shape);
  data->x_norm_->set_shape(in_shape);

  shape[0] = in_shape.shape(0);
  data->temp_->set_shape(shape);

  int spatial_dim = in_shape.count() / (param->channels_ * in_shape.shape(0));
  if (data->spatial_sum_multiplier_->shape().num_axes() == 0 ||
    data->spatial_sum_multiplier_->shape().shape(0) != spatial_dim) {
    shape[0] = spatial_dim;
    data->spatial_sum_multiplier_->set_shape(shape);
    // TODO(depeng): spatial_sum_multiplier_ need to be initialized to 1
#if 0
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
#endif
  }

  int numbychans = param->channels_ * in_shape.shape(0);
  if (data->num_by_chans_->shape().num_axes() == 0 ||
    data->num_by_chans_->shape().shape(0) != numbychans) {
    shape[0] = numbychans;
    data->num_by_chans_->set_shape(shape);
    // TODO(depeng): batch_sum_multiplier_ need to be initialized to 1
#if 0
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
      batch_sum_multiplier_.mutable_cpu_data());
#endif
  }

  // Finally, align the blob shapes in
  // this->param_->prototype_data_ with |data_param|
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}


INSTANTIATE_CLASS(BatchNormLayer);
REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace caffe



#if 0

template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
        this->blobs_[i]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
    spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
    num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
      batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
      0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor,
      this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
      this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    // compute mean
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
      1. / (num * spatial_dim), bottom_data,
      spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());
  }

  // subtract mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
    num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
    spatial_dim, 1, -1, num_by_chans_.cpu_data(),
    spatial_sum_multiplier_.cpu_data(), 1., top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(top[0]->count(), top_data, Dtype(2),
      temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
      1. / (num * spatial_dim), temp_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      variance_.mutable_cpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
      moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
    int m = bottom[0]->count() / channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
    caffe_cpu_axpby(variance_.count(), bias_correction_factor,
      variance_.cpu_data(), moving_average_fraction_,
      this->blobs_[1]->mutable_cpu_data());
  }

  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
    variance_.mutable_cpu_data());

  // replicate variance to input size
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
    num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
    spatial_dim, 1, 1., num_by_chans_.cpu_data(),
    spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data,
    x_norm_.mutable_cpu_data());
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
    top_diff = x_norm_.cpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (use_global_stats_) {
    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are Hadamard product and element-wise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
    bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
    num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
    num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
    mean_.mutable_cpu_data());

  // reshape (broadcast) the above
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
    num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
    spatial_dim, 1, 1., num_by_chans_.cpu_data(),
    spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
    top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
    num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
    num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
    mean_.mutable_cpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
    num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
    spatial_dim, 1, 1., num_by_chans_.cpu_data(),
    spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
    Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}
#endif
