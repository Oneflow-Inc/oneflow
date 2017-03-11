//#ifdef USE_CUDNN
#include <vector>

#include "layers/cudnn_convolution_layer.h"
#include "layers/layer_factory.h"

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(CuDNNConvolutionData, data, data_param);
  GET_CONCRETE_POINTER(ConvolutionModel, model, model_param);
  GET_CONCRETE_POINTER(CuDNNConvolutionParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(model->weight);
  CHECK_NOTNULL(model->weight_diff);
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);

  const Dtype* weight = model->weight->data();
  const Dtype* in_data = data->in->data();
  Dtype* out_data = data->out->mutable_data();

  int i = 0;
  // Forward through cuDNN in parallel over groups.
  for (int g = 0; g < param->group_; g++) {
    // Filters.
    CUDNN_CHECK(cudnnConvolutionForward(ctx.cudnn_handle,
      cudnn::dataType<Dtype>::one,
      param->in_descs_, in_data + param->in_offset_ * g,
      param->filter_desc_, weight + param->weight_offset_ * g,
      param->conv_descs_,
      param->fwd_algo_, NULL, 0,
      cudnn::dataType<Dtype>::zero,
      param->out_descs_, out_data + param->out_offset_ * g));

    // Bias.
    if (param->bias_term_) {
      CHECK_NOTNULL(model->bias);
      CHECK_NOTNULL(model->bias_multiplier);
      const Dtype* bias_data = model->bias->data();

#if CUDNN_VERSION_MIN(4, 0, 0)
      CUDNN_CHECK(cudnnAddTensor(ctx.cudnn_handle,
        cudnn::dataType<Dtype>::one,
        param->bias_desc_, bias_data + param->bias_offset_ * g,
        cudnn::dataType<Dtype>::one,
        param->out_descs_, out_data + param->out_offset_ * g));
#else
      CUDNN_CHECK(cudnnAddTensor(ctx.cudnn_handle, CUDNN_ADD_SAME_C,
        cudnn::dataType<Dtype>::one,
        param->bias_desc_, bias_data + param->bias_offset_ * g,
        cudnn::dataType<Dtype>::one,
        param->out_descs_, out_data + param->out_offset_ * g));
#endif
    }
  }

  // Synchronize the work across groups, each of which went into its own
  // stream, by launching an empty kernel into the default (null) stream.
  // NOLINT_NEXT_LINE(whitespace/operators)
  sync_conv_groups << <1, 1 >> >();
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(CuDNNConvolutionData, data, data_param);
  GET_CONCRETE_POINTER(ConvolutionModel, model, model_param);
  GET_CONCRETE_POINTER(CuDNNConvolutionParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";
  CUDNN_CHECK(cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream));

  CHECK_NOTNULL(model->weight);
  CHECK_NOTNULL(model->weight_diff);
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->out_diff);

  const Dtype* weight_ = model->weight->data();
  const Blob<Dtype>* weight_diff_ = model->weight_diff;
  CUDA_CHECK(cudaMemsetAsync(weight_diff_->mutable_data(), (Dtype)0,
    weight_diff_->shape().count()*sizeof(Dtype), ctx.cuda_stream));

  Blob<Dtype>* bias_diff_ = NULL;
  if (param->bias_term_) {
    CHECK_NOTNULL(model->bias);
    CHECK_NOTNULL(model->bias_diff);
    CHECK_NOTNULL(model->bias_multiplier);

    bias_diff_ = model->bias_diff;
    CUDA_CHECK(cudaMemsetAsync(bias_diff_->mutable_data(), (Dtype)0,
      bias_diff_->shape().count()*sizeof(Dtype), ctx.cuda_stream));

  }
  int i = 0;
  const Dtype* out_diff = data->out_diff->data();
  // Backward through cuDNN in parallel over groups and gradients.
  for (int g = 0; g < param->group_; g++) {
    // Gradient w.r.t. bias.
    if (param->bias_term_) {
      CUDNN_CHECK(cudnnConvolutionBackwardBias(ctx.cudnn_handle,
        cudnn::dataType<Dtype>::one,
        param->out_descs_, out_diff + param->out_offset_ * g,
        cudnn::dataType<Dtype>::one,
        param->bias_desc_, bias_diff_->mutable_data() + param->bias_offset_ * g));
    }

    // Gradient w.r.t. weights.
    const Dtype* in_data = data->in->data();
    CUDNN_CHECK(cudnnConvolutionBackwardFilter_v3(
      ctx.cudnn_handle,
      cudnn::dataType<Dtype>::one,
      param->in_descs_, in_data + param->in_offset_ * g,
      param->out_descs_, out_diff + param->out_offset_ * g,
      param->conv_descs_,
      param->bwd_filter_algo_, NULL, 0,
      cudnn::dataType<Dtype>::one,
      param->filter_desc_, weight_diff_->mutable_data() + param->weight_offset_ * g));

    // Gradient w.r.t. bottom data.

    Dtype* in_diff = data->in_diff->mutable_data();
    CUDNN_CHECK(cudnnConvolutionBackwardData_v3(
      ctx.cudnn_handle,
      cudnn::dataType<Dtype>::one,
      param->filter_desc_, weight_ + param->weight_offset_ * g,
      param->out_descs_, out_diff + param->out_offset_ * g,
      param->conv_descs_,
      param->bwd_data_algo_, NULL, 0,
      cudnn::dataType<Dtype>::zero,
      param->in_descs_, in_diff + param->in_offset_ * g));
  }

  // Synchronize the work across groups, each of which went into its own
  // stream, by launching an empty kernel into the default (null) stream.
  // NOLINT_NEXT_LINE(whitespace/operators)
  sync_conv_groups << <1, 1 >> >();

}

INSTANTIATE_LAYER_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
//#endif
