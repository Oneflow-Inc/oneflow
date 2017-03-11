//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_softmax_layer.h"
#include "layers/layer_factory.h"
namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(SoftmaxData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNSoftmaxParam, param, param_);
  GET_CONCRETE_POINTER(CuDNNSoftmaxModel, model, model_param);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  //CHECK_NOTNULL(data->out_copy);
  //CHECK_NOTNULL(model->scale);

  const Dtype* in_data = data->in->data();
  Dtype* out_data = data->out->mutable_data();
  // Dtype* out_copy_data = data->out_copy->mutable_data();
  //Dtype* scale_data = model->scale->mutable_data();
  int count = data->out->shape().count();
  //int channels = data->out->shape().shape(param->softmax_axis_);

  CUDNN_CHECK(cudnnSoftmaxForward(ctx.cudnn_handle, CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_MODE_CHANNEL,
    cudnn::dataType<Dtype>::one,
    param->in_desc_, in_data,
    cudnn::dataType<Dtype>::zero,
    param->out_desc_, out_data));
#if 0  
  if (ctx.cuda_stream) {
    CUDA_CHECK(cudaMemcpyAsync(out_copy_data,
      data->out->data(), count*sizeof(Dtype),
      cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  }
  else {
    CUDA_CHECK(cudaMemcpy(out_copy_data,
      data->out->data(), count*sizeof(Dtype),
      cudaMemcpyDeviceToDevice));
  }
#endif 
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(SoftmaxData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNSoftmaxParam, param, param_);
  GET_CONCRETE_POINTER(CuDNNSoftmaxModel, model, model_param);
  
  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  // Use ctx, data and model
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->out_diff);
  CHECK_NOTNULL(data->in_diff);

  //CHECK_NOTNULL(model->scale);
  const Dtype* out_data = data->out->data();
  const Dtype* out_diff = data->out_diff->data();
  Dtype* in_diff = data->in_diff->mutable_data();
  //Dtype* scale_data = model->scale->mutable_data();
  //int count = data->out->shape().count();
  //int channels = data->out->shape().shape(param->softmax_axis_);
  
  CUDNN_CHECK(cudnnSoftmaxBackward(ctx.cudnn_handle, CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_MODE_CHANNEL,
    cudnn::dataType<Dtype>::one,
    param->out_desc_, out_data, param->out_desc_, out_diff,
    cudnn::dataType<Dtype>::zero,
    param->in_desc_, in_diff));
}

INSTANTIATE_LAYER_FUNCS(CuDNNSoftmaxLayer);

}  // namespace caffe
//#endif