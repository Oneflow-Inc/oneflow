#include <string>
#include <vector>

#include "memory/blob.h"
#include "common/common.h"
#include "layers/innerproduct_layer.h"
#include "math/math_util.h"

namespace caffe {
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(InnerProductData, data, data_param);
  GET_CONCRETE_POINTER(InnerProductModel, model, model_param);
  GET_CONCRETE_POINTER(InnerProductParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(model->weight);
  CHECK_NOTNULL(model->weight_diff);
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasTrans,
    param->num_example_,
    param->num_output_,
    param->num_input_, (Dtype)1., data->in->data(),
    model->weight->data(), (Dtype)0., data->out->mutable_data(),
    ctx.cuda_stream);
  if (param->bias_term_) {
    CHECK_NOTNULL(model->bias);
    CHECK_NOTNULL(model->bias_multiplier);
    caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans,
      param->num_example_, param->num_output_, 1, (Dtype)1.,
      model->bias_multiplier->data(), model->bias->data(),
      (Dtype)1., data->out->mutable_data(), ctx.cuda_stream);
  }
}
template <typename Dtype>
void InnerProductLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(InnerProductData, data, data_param);
  GET_CONCRETE_POINTER(InnerProductModel, model, model_param);
  GET_CONCRETE_POINTER(InnerProductParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(model->weight);
  CHECK_NOTNULL(model->weight_diff);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);
  // Gradient with respect to weight
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasTrans, CblasNoTrans,
    param->num_output_,
    param->num_input_,
    param->num_example_, (Dtype)1., data->out_diff->data(),
    data->in->data(), (Dtype)0., model->weight_diff->mutable_data(),
    ctx.cuda_stream);
  if (param->bias_term_) {
    CHECK_NOTNULL(model->bias);
    CHECK_NOTNULL(model->bias_diff);
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(ctx.cublas_handle, CblasTrans, param->num_example_,
      param->num_output_,
      (Dtype)1., data->out_diff->data(),
      model->bias_multiplier->data(), (Dtype)0.,
      model->bias_diff->mutable_data(), ctx.cuda_stream);
  }
  // Gradient with respect to bottom data
  caffe_gpu_gemm<Dtype>(ctx.cublas_handle, CblasNoTrans, CblasNoTrans,
    param->num_example_, param->num_input_, param->num_output_, (Dtype)1.,
    data->out_diff->data(), model->weight->data(),
    (Dtype)0., data->in_diff->mutable_data(), ctx.cuda_stream);
}
INSTANTIATE_LAYER_FUNCS(InnerProductLayer);
#if 0
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {
  CHECK(model->find("weight") != model->end());
  const Blob<Dtype>* weight_ = model->find("weight")->second.get();
  const Blob<Dtype>* bias_ = nullptr;
  const Blob<Dtype>* bias_multiplier_ = nullptr;
  const Dtype* inputs_data = inputs[0]->data();
  Dtype* outputs_data = (*outputs)[0]->mutable_data();
  caffe_gpu_gemm<Dtype>(cublas_handle, CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., inputs_data, weight_->data(), (Dtype)0.,
    outputs_data, cuda_stream);
  if (bias_term_) {
    CHECK(model->find("bias") != model->end());
    CHECK(model->find("bias_multiplier") != model->end());
    bias_ = model->find("bias")->second.get();
    bias_multiplier_ = model->find("bias_multiplier")->second.get();
    caffe_gpu_gemm<Dtype>(cublas_handle, CblasNoTrans, CblasNoTrans, M_, N_, 1,
    (Dtype)1., bias_multiplier_->data(), bias_->data(),
    (Dtype)1., outputs_data, cuda_stream);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {
  CHECK(model->find("weight") != model->end());
  CHECK(model->find("weight_diff") != model->end());
  const Blob<Dtype>* weight_ = model->find("weight")->second.get();
  // TODO(xcdu) :2015.10.16 get the pointer of weight_diff blob.
  const Blob<Dtype>* weight_diff_ = model->find("weight_diff")->second.get();
  const Blob<Dtype>* bias_diff_ = nullptr;
  const Blob<Dtype>* bias_multiplier_ = nullptr;
  // loss stores in outputs_data blobs when backward.
  const Dtype* inputs_data_ = (*inputs)[0]->data();
  const Dtype* outputs_diff_ = outputs[0]->data();
  // Gradient with respect to weight
  caffe_gpu_gemm<Dtype>(cublas_handle, CblasTrans, CblasNoTrans, N_, K_, M_,
    (Dtype)1., outputs_diff_, inputs_data_, (Dtype)0.,
    weight_diff_->mutable_data(), cuda_stream);
  if (bias_term_) {
    CHECK(model->find("bias_diff") != model->end());
    CHECK(model->find("bias_multiplier") != model->end());
    bias_diff_ = model->find("bias_diff")->second.get();
    bias_multiplier_ = model->find("bias_multiplier")->second.get();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(cublas_handle, CblasTrans, M_, N_, (Dtype)1.,
      outputs_diff_, bias_multiplier_->data(), (Dtype)0.,
      bias_diff_->mutable_data(), cuda_stream);
  }
  // Gradient with respect to bottom data
  caffe_gpu_gemm<Dtype>(cublas_handle, CblasNoTrans, CblasNoTrans, M_, K_, N_,
    (Dtype)1., outputs_diff_, weight_->data(), (Dtype)0.,
    (*inputs)[0]->mutable_data(), cuda_stream);
}
#endif
}  // namespace caffe
