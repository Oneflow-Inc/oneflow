#include <vector>
#include "layers/batch_norm_layer.h"
#include "math/math_util.h"
#include "memory/blob_util.h"


namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(BatchNormData, data, data_param);
  GET_CONCRETE_POINTER(BatchNormModel, model, model_param);
  GET_CONCRETE_POINTER(BatchNormParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  for (int i = 0; i<model->blobs_.size(); ++i){
    CHECK_NOTNULL(model->blobs_[i]);
  }
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->mean_);
  CHECK_NOTNULL(data->temp_);
  CHECK_NOTNULL(data->variance_);
  CHECK_NOTNULL(data->x_norm_);
  CHECK_NOTNULL(data->num_by_chans_);
  CHECK_NOTNULL(data->batch_sum_multiplier_);
  CHECK_NOTNULL(data->spatial_sum_multiplier_);

  const Dtype* in_data = data->in->data();
  Dtype* out_data = data->out->mutable_data();

  const Shape& in_shape = data->in->shape();
  int num = in_shape.shape(0);
  int spatial_dim = in_shape.count() / (param->channels_* num);
  if (data->in != data->out) {
    caffe_gpu_async_copy(in_shape.count(), in_data, out_data, ctx.cuda_stream);
  }

  if (param->use_global_stats_) {

    Blob<Dtype>* model_cpu_blobs = new Blob<Dtype>();
    model_cpu_blobs->set_shape(model->blobs_[2]->shape());
    model_cpu_blobs->Alloc(DeviceType::kCPUPinned);
    // NOTE(depeng): not compiled
    // AyncCopyD2H(*(model->blobs_[2]), model_cpu_blobs, ctx.cuda_stream);
    CUDA_CHECK(cudaMemcpy(model_cpu_blobs->mutable_data(),
      model->blobs_[2]->data(), model->blobs_[2]->byte_size(),
      cudaMemcpyDeviceToHost));

    // use the stored mean/variance estimates.
    const Dtype scale_factor = model_cpu_blobs->data()[0] == 0 ?
      0 : 1 / model_cpu_blobs->data()[0];

    caffe_gpu_scale<Dtype>(ctx.cublas_handle, data->variance_->shape().count(),
      scale_factor, model->blobs_[0]->data(), data->mean_->mutable_data(),
      ctx.cuda_stream);

    caffe_gpu_scale<Dtype>(ctx.cublas_handle, data->variance_->shape().count(),
      scale_factor, model->blobs_[1]->data(), data->variance_->mutable_data(),
      ctx.cuda_stream);

    delete model_cpu_blobs;
  } else {
    // compute mean
    caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasNoTrans, param->channels_ * num,
      spatial_dim, 1. / (num * spatial_dim), in_data,
      data->spatial_sum_multiplier_->data(), 0.0,
      data->num_by_chans_->mutable_data(), ctx.cuda_stream);

    caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasTrans, num, param->channels_,
      1., data->num_by_chans_->data(), data->batch_sum_multiplier_->data(), 0.0,
      data->mean_->mutable_data(), ctx.cuda_stream);
  }

  // subtract mean
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans,
    num, param->channels_, 1, 1, data->batch_sum_multiplier_->data(),
    data->mean_->data(), 0., data->num_by_chans_->mutable_data(),
    ctx.cuda_stream);
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans,
    param->channels_ * num, spatial_dim, 1, -1, data->num_by_chans_->data(),
    data->spatial_sum_multiplier_->data(), 1., out_data,
    ctx.cuda_stream);


  if (!param->use_global_stats_) {

    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx(in_shape.count(), out_data, Dtype(2),
      data->temp_->mutable_data(), ctx.cuda_stream);  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasNoTrans,
      param->channels_ * num, spatial_dim, 1. / (num * spatial_dim),
      data->temp_->data(), data->spatial_sum_multiplier_->data(), 0.,
      data->num_by_chans_->mutable_data(), ctx.cuda_stream);

    caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasTrans, num,
      param->channels_, 1., data->num_by_chans_->data(),
      data->batch_sum_multiplier_->data(), 0.,
      data->variance_->mutable_data(), ctx.cuda_stream);  // E((X_EX)^2)

    // model->blobs_[2]->mutable_data()[0] *= param->moving_average_fraction_;
    caffe_gpu_scal<Dtype>(ctx.cublas_handle, model->blobs_[2]->shape().count(),
      param->moving_average_fraction_, model->blobs_[2]->mutable_data(),
      ctx.cuda_stream);
    // compute and save moving average
    // model->blobs_[2]->mutable_data()[0] += 1;
    caffe_gpu_add_scalar<Dtype>(model->blobs_[2]->shape().count(),
      (Dtype)1, model->blobs_[2]->mutable_data(), ctx.cuda_stream);

    caffe_gpu_axpby<Dtype>(ctx.cublas_handle, data->mean_->shape().count(),
      Dtype(1), data->mean_->data(), param->moving_average_fraction_,
      model->blobs_[0]->mutable_data(), ctx.cuda_stream);

    int m = in_shape.count() / param->channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
    caffe_gpu_axpby<Dtype>(ctx.cublas_handle, data->variance_->shape().count(),
      bias_correction_factor, data->variance_->data(),
      param->moving_average_fraction_, model->blobs_[1]->mutable_data(),
      ctx.cuda_stream);
  }


  // normalize variance
  caffe_gpu_add_scalar<Dtype>(data->variance_->shape().count(), param->eps_,
    data->variance_->mutable_data(), ctx.cuda_stream);
  caffe_gpu_powx<Dtype>(data->variance_->shape().count(),
    data->variance_->data(), Dtype(0.5),
    data->variance_->mutable_data(), ctx.cuda_stream);

  // replicate variance to input size
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans, num,
    param->channels_, 1, 1, data->batch_sum_multiplier_->data(),
    data->variance_->data(), 0., data->num_by_chans_->mutable_data(),
    ctx.cuda_stream);
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans,
    param->channels_ * num, spatial_dim, 1, 1., data->num_by_chans_->data(),
    data->spatial_sum_multiplier_->data(), 0., data->temp_->mutable_data(),
    ctx.cuda_stream);

  caffe_gpu_div<Dtype>(data->temp_->shape().count(), out_data,
    data->temp_->data(), out_data, ctx.cuda_stream);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  // might clobber the data.  Can we skip this if they won't?
  caffe_gpu_async_copy<Dtype>(data->x_norm_->shape().count(), out_data,
    data->x_norm_->mutable_data(), ctx.cuda_stream);

}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(BatchNormData, data, data_param);
  GET_CONCRETE_POINTER(BatchNormModel, model, model_param);
  GET_CONCRETE_POINTER(BatchNormParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  for (int i = 0; i<model->blobs_.size(); ++i){
    CHECK_NOTNULL(model->blobs_[i]);
  }
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);
  CHECK_NOTNULL(data->mean_);
  CHECK_NOTNULL(data->temp_);
  CHECK_NOTNULL(data->variance_);
  CHECK_NOTNULL(data->x_norm_);
  CHECK_NOTNULL(data->num_by_chans_);
  CHECK_NOTNULL(data->batch_sum_multiplier_);
  CHECK_NOTNULL(data->spatial_sum_multiplier_);

  const Dtype* out_diff = data->out_diff->data();
  /*
  if (data->in != data->out) {
  out_diff = data->out->data();
  } else {
  // caffe_copy(x_norm_.count(), top[0]->gpu_diff(), x_norm_.mutable_gpu_diff());
  CUDA_CHECK(cudaMemcpy(data->x_norm_->mutable_data(), data->out->data(),
  sizeof(Dtype) * data->x_norm_->shape().count(), cudaMemcpyDeviceToDevice));
  out_diff = data->x_norm_->data();
  }
  */

  Dtype* in_diff = data->in_diff->mutable_data();

  if (param->use_global_stats_) {
    caffe_gpu_div<Dtype>(data->temp_->shape().count(), out_diff,
      data->temp_->data(), in_diff, ctx.cuda_stream);
    return;
  }

  const Dtype* out_data = data->x_norm_->data();
  const Shape& in_shape = data->in->shape();

  int num = in_shape.num();
  int spatial_dim = in_shape.count() / (param->channels_ * num);
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
  // caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_gpu_mul<Dtype>(data->temp_->shape().count(), out_data,
    out_diff, in_diff, ctx.cuda_stream);

  caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasNoTrans, param->channels_ * num,
    spatial_dim, 1., in_diff, data->spatial_sum_multiplier_->data(), 0.,
    data->num_by_chans_->mutable_data(), ctx.cuda_stream);

  caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasTrans, num,
    param->channels_, 1., data->num_by_chans_->data(),
    data->batch_sum_multiplier_->data(), 0.,
    data->mean_->mutable_data(), ctx.cuda_stream);

  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans, num,
    param->channels_, 1, 1, data->batch_sum_multiplier_->data(),
    data->mean_->data(), 0., data->num_by_chans_->mutable_data(), ctx.cuda_stream);
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans,
    param->channels_ * num, spatial_dim, 1, 1., data->num_by_chans_->data(),
    data->spatial_sum_multiplier_->data(), 0., in_diff, ctx.cuda_stream);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul<Dtype>(data->temp_->shape().count(), out_data,
    in_diff, in_diff, ctx.cuda_stream);


  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasNoTrans, param->channels_ * num,
    spatial_dim, 1., out_diff, data->spatial_sum_multiplier_->data(), 0.,
    data->num_by_chans_->mutable_data(), ctx.cuda_stream);
  caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasTrans, num, param->channels_,
    1., data->num_by_chans_->data(), data->batch_sum_multiplier_->data(), 0.,
    data->mean_->mutable_data(), ctx.cuda_stream);

  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans, num,
    param->channels_, 1, 1, data->batch_sum_multiplier_->data(), data->mean_->data(),
    0., data->num_by_chans_->mutable_data(), ctx.cuda_stream);
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans, num * param->channels_,
    spatial_dim, 1, 1., data->num_by_chans_->data(), data->spatial_sum_multiplier_->data(),
    1., in_diff, ctx.cuda_stream);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby<Dtype>(ctx.cublas_handle, data->temp_->shape().count(), Dtype(1),
    out_diff, Dtype(-1. / (num * spatial_dim)), in_diff, ctx.cuda_stream);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_gpu_div<Dtype>(data->temp_->shape().count(), in_diff, data->temp_->data(),
    in_diff, ctx.cuda_stream);
}

INSTANTIATE_LAYER_FUNCS(BatchNormLayer);

};  // namespace caffe