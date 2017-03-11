//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_relu_layer.h"
#include "layers/layer_factory.h"
namespace caffe {

template <typename Dtype>
void CuDNNReLULayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ReLUData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNReLUParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  const Dtype* in_data = data->in->data();
  Dtype* out_data = data->out->mutable_data();

  CUDNN_CHECK(cudnnActivationForward(ctx.cudnn_handle,
    CUDNN_ACTIVATION_RELU,
    cudnn::dataType<Dtype>::one,
    param->in_desc_, in_data,
    cudnn::dataType<Dtype>::zero,
    param->out_desc_, out_data));
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  
  GET_CONCRETE_POINTER(ReLUData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNReLUParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);

  const Dtype* out_diff_ = data->out_diff->data();
  const Dtype* in_data_ = data->in->data();
  Dtype* in_diff_ = data->in_diff->mutable_data();

  CUDNN_CHECK(cudnnActivationBackward(ctx.cudnn_handle,
    CUDNN_ACTIVATION_RELU,
    cudnn::dataType<Dtype>::one,
    param->out_desc_, data->out->data(), param->out_desc_, out_diff_,
    param->in_desc_, in_data_,
    cudnn::dataType<Dtype>::zero,
    param->in_desc_, in_diff_));
}

INSTANTIATE_LAYER_FUNCS(CuDNNReLULayer);

}  // namespace caffe
//#endif
