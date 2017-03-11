//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_pooling_layer.h"
#include "layers/layer_factory.h"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(CuDNNPoolingData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNPoolingParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  // CHECK_NOTNULL(data->idx);
  const Dtype* in_data = data->in->data();
  Dtype* out_data = data->out->mutable_data();
  int count = data->out->shape().count();

  CUDNN_CHECK(cudnnPoolingForward(ctx.cudnn_handle, param->pooling_desc_,
    cudnn::dataType<Dtype>::one,
    param->in_desc_, in_data,
    cudnn::dataType<Dtype>::zero,
    param->out_desc_, out_data));

  // if (ctx.cuda_stream) {
  //   CUDA_CHECK(cudaMemcpyAsync(data->idx->mutable_data(),
  //     data->out->data(), count*sizeof(Dtype),
  //     cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  // } else {
  //   CUDA_CHECK(cudaMemcpy(data->idx->mutable_data(),
  //     data->out->data(), count*sizeof(Dtype),
  //     cudaMemcpyDeviceToDevice));
  // }
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(CuDNNPoolingData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNPoolingParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  // Use ctx, data and model
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);
  // CHECK_NOTNULL(data->idx);
  const Dtype* out_diff_ = data->out_diff->data();
  Dtype* in_diff_ = data->in_diff->mutable_data();

  // Dtype* inputs_gpu_data;
  // CUDA_CHECK(cudaMalloc(&inputs_gpu_data, data->in->shape().count()*sizeof(Dtype)));
  // CUDA_CHECK(cudaMemcpy(inputs_gpu_data, data->in->data(),
  //   data->in->shape().count() * sizeof(Dtype),
  //   cudaMemcpyDeviceToDevice));

  CUDNN_CHECK(cudnnPoolingBackward(ctx.cudnn_handle, param->pooling_desc_,
    cudnn::dataType<Dtype>::one,
    param->out_desc_, data->out->data(),
    param->out_desc_, out_diff_,
    param->in_desc_, data->in->data(),
    cudnn::dataType<Dtype>::zero,
    param->in_desc_, in_diff_));
}

INSTANTIATE_LAYER_FUNCS(CuDNNPoolingLayer);

}  // namespace caffe
//#endif