//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_lrn_layer.h"

namespace caffe {

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(LRNData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNLRNParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);

  const Dtype* in_data = data->in->data();
  Dtype* out_data = data->out->mutable_data();


  CUDNN_CHECK(cudnnLRNCrossChannelForward(
    ctx.cudnn_handle, param->norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
    cudnn::dataType<Dtype>::one,
    param->in_desc_, in_data,
    cudnn::dataType<Dtype>::zero,
    param->in_desc_, out_data));

  // if (ctx.cuda_stream) {
  //   CUDA_CHECK(cudaMemcpyAsync(data->in_copy->mutable_data(),
  //     data->in->data(), data->in->shape().count()*sizeof(Dtype),
  //     cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  //   CUDA_CHECK(cudaMemcpyAsync(data->out_copy->mutable_data(),
  //     data->out->data(), data->out->shape().count()*sizeof(Dtype),
  //     cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  // }
  // else {
  //   CUDA_CHECK(cudaMemcpy(data->in_copy->mutable_data(),
  //     data->in->data(), data->in->shape().count()*sizeof(Dtype),
  //     cudaMemcpyDeviceToDevice));
  //   CUDA_CHECK(cudaMemcpy(data->out_copy->mutable_data(),
  //     data->out->data(), data->out->shape().count()*sizeof(Dtype),
  //     cudaMemcpyDeviceToDevice));
  // }
}

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(LRNData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNLRNParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);

  const Dtype* out_diff = data->out_diff->data();
  const Dtype* out_data = data->out->data();
  const Dtype* in_data = data->in->data();
  Dtype* in_diff = data->in_diff->mutable_data();

  CUDNN_CHECK(cudnnLRNCrossChannelBackward(
    ctx.cudnn_handle, param->norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
    cudnn::dataType<Dtype>::one,
    param->in_desc_, out_data,
    param->in_desc_, out_diff,
    param->out_desc_, in_data,
    cudnn::dataType<Dtype>::zero,
    param->out_desc_, in_diff));
}

INSTANTIATE_LAYER_FUNCS(CuDNNLRNLayer);

};  // namespace caffe

//#endif
