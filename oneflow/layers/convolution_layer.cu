#include <string>
#include <vector>

#include "common/common.h"
#include "layers/convolution_layer.h"
#include "math/math_util.h"

// NOTE(xcdu): param_propagate_down_ is a parameter to control
#define param_propagate_down_ 1

namespace caffe {
template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_gpu_gemm(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const Dtype* input, const Dtype* weights,
  const Blob<Dtype>* col_buffer_, Dtype* output,
  bool skip_im2col) const {
  GET_CONCRETE_POINTER(ConvolutionParam, param, param_);
  const Dtype* col_buff = input;
  if (!param->is_1x1_) {
    if (!skip_im2col) {
    conv_im2col_gpu(cuda_stream, input, col_buffer_->mutable_data());
    }
    col_buff = col_buffer_->data();
  }
  for (int g = 0; g < param->group_; ++g) {
    caffe_gpu_gemm<Dtype>(cublas_handle, CblasNoTrans, CblasNoTrans,
    param->out_channels_ /param->group_, param->conv_out_spatial_dim_,
    param->kernel_dim_ / param->group_,
    (Dtype)1., weights + param->weight_offset_ * g, col_buff +
    param->col_offset_ * g, (Dtype)0., output + param->output_offset_ * g,
    cuda_stream);
  }
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_gpu_bias(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const Dtype* bias, const Dtype* bias_multiplier_, Dtype* output) const {
  GET_CONCRETE_POINTER(ConvolutionParam, param, param_);
  caffe_gpu_gemm<Dtype>(cublas_handle, CblasNoTrans, CblasNoTrans,
    param->out_channels_, param->out_height_ * param->out_width_, (Dtype)1,
    (Dtype)1., bias, bias_multiplier_, (Dtype)1., output, cuda_stream);
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::weight_gpu_gemm(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const Dtype* input, const Dtype* output,
  const Blob<Dtype>* col_buffer_, Dtype* weights) const {
  GET_CONCRETE_POINTER(ConvolutionParam, param, param_);
  const Dtype* col_buff = input;
  if (!param->is_1x1_) {
    conv_im2col_gpu(cuda_stream, input, col_buffer_->mutable_data());
    col_buff = col_buffer_->data();
  }
  for (int g = 0; g < param->group_; ++g) {
    caffe_gpu_gemm<Dtype>(cublas_handle, CblasNoTrans, CblasTrans,
      param->out_channels_ / param->group_, param->kernel_dim_ /
      param->group_, param->conv_out_spatial_dim_, (Dtype)1., 
      output + param->output_offset_ * g, col_buff + param->col_offset_ * g,
      (Dtype)1., weights + param->weight_offset_ * g, cuda_stream);
  }
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::backward_gpu_gemm(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const Dtype* output, const Dtype* weights, const Blob<Dtype>* col_buffer_,
  Dtype* input) const {
  GET_CONCRETE_POINTER(ConvolutionParam, param, param_);
  Dtype* col_buff = col_buffer_->mutable_data();
  if (param->is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < param->group_; ++g) {
    caffe_gpu_gemm<Dtype>(cublas_handle, CblasTrans, CblasNoTrans,
      param->kernel_dim_ / param->group_, param->conv_out_spatial_dim_,
      param->out_channels_ / param->group_,
      (Dtype)1., weights + param->weight_offset_ * g,
      output + param->output_offset_ * g,
      (Dtype)0., col_buff + param->col_offset_ * g, cuda_stream);
  }
  if (!param->is_1x1_) {
    conv_col2im_gpu(cuda_stream, col_buff, input);
  }
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ConvolutionData, data, data_param);
  GET_CONCRETE_POINTER(ConvolutionModel, model, model_param);
  GET_CONCRETE_POINTER(ConvolutionParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(model->weight);
  CHECK_NOTNULL(model->weight_diff);
  CHECK_NOTNULL(data->col_buf);
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  const Dtype* inputs_data = data->in->data();
  Dtype* outputs_data = data->out->mutable_data();
  for (int n = 0; n < param->num_; ++n) {
    forward_gpu_gemm(ctx.cublas_handle, ctx.cuda_stream,
      inputs_data + data->in->shape().offset(n),
      model->weight->data(), data->col_buf, outputs_data +
      data->out->shape().offset(n),
      false);

    if (param->bias_term_) {
      CHECK_NOTNULL(model->bias);
      CHECK_NOTNULL(model->bias_multiplier);
      const Dtype* bias_ = model->bias->data();
      const Dtype* bias_multiplier_ =
        model->bias_multiplier->data();
      forward_gpu_bias(ctx.cublas_handle, ctx.cuda_stream, bias_,
        bias_multiplier_, outputs_data + data->out->shape().offset(n));
    }
  }
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ConvolutionData, data, data_param);
  GET_CONCRETE_POINTER(ConvolutionModel, model, model_param);
  GET_CONCRETE_POINTER(ConvolutionParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(model->weight);
  CHECK_NOTNULL(model->weight_diff);
  CHECK_NOTNULL(data->col_buf);
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);

  const Dtype* weight_ = model->weight->data();
  const Blob<Dtype>* weight_diff_ = model->weight_diff;
  const Blob<Dtype>* col_buffer_ = data->col_buf;

  // NOTE(jiyuan): use async version instead
  // CUDA_CHECK(cudaMemset(weight_diff_->mutable_data(), (Dtype)0,
  //   weight_diff_->shape().count()*sizeof(Dtype)));
  CUDA_CHECK(cudaMemsetAsync(weight_diff_->mutable_data(), (Dtype)0,
    weight_diff_->shape().count()*sizeof(Dtype), ctx.cuda_stream));

  const Blob<Dtype>* bias_multiplier_ = nullptr;
  const Blob<Dtype>* bias_diff_ = nullptr;
  if (param->bias_term_) {
    CHECK_NOTNULL(model->bias);
    CHECK_NOTNULL(model->bias_diff);
    CHECK_NOTNULL(model->bias_multiplier);

    bias_diff_ = model->bias_diff;
    bias_multiplier_ = model->bias_multiplier;
    CUDA_CHECK(cudaMemsetAsync(bias_diff_->mutable_data(), (Dtype)0,
      bias_diff_->shape().count()*sizeof(Dtype), ctx.cuda_stream));
  }
  // NOTE(jiyuan): in backward, we assume in/in_diff, out/out_diff use the same
  // memory blob.
  const Dtype* out_diff_ = data->out_diff->data();
  // Bias gradient, if necessary.
  if (param->bias_term_) {
    for (int n = 0; n < param->num_; ++n) {
      caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasNoTrans, param->out_channels_,
        param->out_height_ * param->out_width_, (Dtype)1.,
        out_diff_ + data->out_diff->shape().offset(n),
        bias_multiplier_->data(), (Dtype)1.0,
        bias_diff_->mutable_data(), ctx.cuda_stream);
    }
  }
  // loss stores in outputs_data blobs when backward.
  const Dtype* input_data = data->in->data();
  Dtype* in_diff = data->in_diff->mutable_data();
  for (int n = 0; n < param->num_; ++n) {
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    weight_gpu_gemm(ctx.cublas_handle, ctx.cuda_stream,
      input_data + data->in->shape().offset(n),
      out_diff_ + data->out_diff->shape().offset(n), col_buffer_,
      weight_diff_->mutable_data());
    // gradient w.r.t. bottom data, if necessary.
    backward_gpu_gemm(ctx.cublas_handle, ctx.cuda_stream,
      out_diff_ + data->out_diff->shape().offset(n), weight_, col_buffer_,
      in_diff + data->in_diff->shape().offset(n));
  }
}
INSTANTIATE_LAYER_FUNCS(ConvolutionLayer);
#if 0
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {
  CHECK(model->find("weight") != model->end());
  CHECK(model->find("col_buffer") != model->end());
  const Blob<Dtype>* weight_ = model->find("weight")->second.get();
  const Blob<Dtype>* col_buffer_ = model->find("col_buffer")->second.get();
  for (int i = 0; i < inputs.size(); ++i) {
    const Dtype* inputs_data = inputs[i]->data();
    Dtype* outputs_data = (*outputs)[i]->mutable_data();
    for (int n = 0; n < num_; ++n) {
      forward_gpu_gemm(cublas_handle, cuda_stream,
        inputs_data + inputs[i]->shape().offset(n),
        weight_->data(), col_buffer_, outputs_data +
        (*outputs)[i]->shape().offset(n),
        false);
      if (bias_term_) {
        CHECK(model->find("bias") != model->end());
        CHECK(model->find("bias_multiplier") != model->end());
        const Dtype* bias_ = model->find("bias")->second->data();
        const Dtype* bias_multiplier_ =
          model->find("bias_multiplier")->second->data();
        forward_gpu_bias(cublas_handle, cuda_stream, bias_, bias_multiplier_,
          outputs_data + (*outputs)[i]->shape().offset(n));
      }
    }
}
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {
  CHECK(model->find("weight") != model->end());
  CHECK(model->find("weight_diff") != model->end());
  CHECK(model->find("col_buffer") != model->end());
  const Dtype* weight_ = model->find("weight")->second->data();
  // TODO(xcdu) :2015.10.16 get the pointer of weight_diff blob.
  const Blob<Dtype>* weight_diff_ = model->find("weight_diff")->second.get();
  const Blob<Dtype>* col_buffer_ = model->find("col_buffer")->second.get();
  const Blob<Dtype>* bias_multiplier_ = nullptr;
  const Blob<Dtype>* bias_diff_ = nullptr;
  CUDA_CHECK(cudaMemset(weight_diff_->mutable_data(), (Dtype)0,
    weight_diff_->shape().count()*sizeof(Dtype)));
  if (bias_term_) {
    CHECK(model->find("bias_diff") != model->end());
    CHECK(model->find("bias_multiplier") != model->end());
    bias_diff_ = model->find("bias_diff")->second.get();
    bias_multiplier_ = model->find("bias_multiplier")->second.get();
    CUDA_CHECK(cudaMemset(bias_diff_->mutable_data(), (Dtype)0,
      bias_diff_->shape().count()*sizeof(Dtype)));
  }
  for (int i = 0; i < (*inputs).size(); ++i) {
    const Dtype* outputs_diff_ = outputs[0]->data();
    // Bias gradient, if necessary.
    if (bias_term_) {
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(cublas_handle, CblasNoTrans, num_output_,
          height_out_*width_out_, (Dtype)1.,
          outputs_diff_+ outputs[0]->shape().offset(n),
          bias_multiplier_->data(), (Dtype)1.0,
          bias_diff_->mutable_data(), cuda_stream);
      }
    }
    // loss stores in outputs_data blobs when backward.
    const Dtype* inputs_data = (*inputs)[i]->data();
    Dtype* inputs_diff = (*inputs)[i]->mutable_data();
    for (int n = 0; n < num_; ++n) {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      weight_gpu_gemm(cublas_handle, cuda_stream,
        inputs_data + (*inputs)[i]->shape().offset(n),
        outputs_diff_ + outputs[i]->shape().offset(n), col_buffer_,
        weight_diff_->mutable_data());
      // gradient w.r.t. bottom data, if necessary.
      backward_gpu_gemm(cublas_handle, cuda_stream,
        outputs_diff_ + outputs[i]->shape().offset(n), weight_, col_buffer_,
        inputs_diff + (*inputs)[i]->shape().offset(n));
    }
  }
}
#endif

}  // namespace caffe
