#include <cuda.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "common/common.h"
#include "layers/multinomial_logistic_loss_layer.h"
#include "math/math_util.h"

template <typename Dtype>
__global__ void mll_layer_select(const int num, const int dim,
  const Dtype kLOG_THRESHOLD, const Dtype* data, const Dtype* label,
  Dtype* loss_buffer) {
  CUDA_KERNEL_LOOP(i, num*dim) {
    loss_buffer[i] = ((static_cast<int>(label[i / dim] == i%dim)) ?
      (-log(max(data[i], (Dtype)kLOG_THRESHOLD))/num): Dtype(0.));
  }
}
namespace caffe {
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(MultinomialLogisticLossData, data, data_param);
  GET_CONCRETE_POINTER(MultinomialLogisticLossModel, model, model_param);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(data->data);
  CHECK_NOTNULL(data->label);
  CHECK_NOTNULL(data->loss);
  CHECK_NOTNULL(data->loss_buffer);
  CHECK_NOTNULL(model->loss_multiplier);
  const Dtype* inputs_data = data->data->data();
  const Dtype*  inputs_label = data->label->data();
  Dtype* outputs_data = data->loss->mutable_data();
  Dtype* loss_buffer = data->loss_buffer->mutable_data();
  Dtype* loss_multipiler = model->loss_multiplier->mutable_data();
  int num = data->data->shape().num();
  int count = data->data->shape().count();
  int out_count = data->loss->shape().count();
  mll_layer_select<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream >> >(num, count / num, kLOG_THRESHOLD,
    inputs_data, inputs_label, loss_buffer);
  caffe_gpu_dot(ctx.cublas_handle, count, loss_buffer, loss_multipiler,
    outputs_data, ctx.cuda_stream);
}
template <typename Dtype>
__global__ void mll_layer_backward(const int num, const int dim,
  const Dtype kLOG_THRESHOLD, const Dtype* data, const Dtype* inputs_label,
  const Dtype* output, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, num*dim) {
    if (i == ((i / dim)*dim + static_cast<int>(inputs_label[i / dim])))
      // (-output[0]/num) is scale;
      diff[i] = - (output[0]/ num) / max(data[i], Dtype(kLOG_THRESHOLD));
    else
      diff[i] = 0;
  }
}
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(MultinomialLogisticLossData, data, data_param);
  GET_CONCRETE_POINTER(MultinomialLogisticLossModel, model, model_param);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  // Use ctx, data and model
  const Dtype* inputs_data = data->data->data();
  const Dtype* inputs_label = data->label->data();
  const Dtype* outputs_data = data->loss->data();
  Dtype* inputs_diff = data->data_diff->mutable_data();
  int num = data->data->shape().num();
  int count = data->data->shape().count();
  mll_layer_backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream>>>(num, count/num, kLOG_THRESHOLD, inputs_data,
    inputs_label, outputs_data, inputs_diff);
}
INSTANTIATE_LAYER_FUNCS(MultinomialLogisticLossLayer);
#if 0
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Forward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {
  CHECK(model->find("loss_multiplier") != model->end());
  CHECK(model->find("loss_buffer") != model->end());
  const Dtype* inputs_data = inputs[0]->data();
  const Dtype*  inputs_label = inputs[1]->data();
  Dtype* outputs_data = (*outputs)[0]->mutable_data();
  Dtype* loss_buffer = model->find("loss_buffer")->second->mutable_data();
  Dtype* loss_multipiler =
    model->find("loss_multiplier")->second->mutable_data();
  int num = inputs[0]->shape().num();
  int count = inputs[0]->shape().count();
  int out_count = (*outputs)[0]->shape().count();
  mll_layer_select<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, cuda_stream >> >(num, count / num, kLOG_THRESHOLD,
    inputs_data, inputs_label, loss_buffer);
  caffe_gpu_dot(cublas_handle, count, loss_buffer, loss_multipiler,
    outputs_data, cuda_stream);
}


template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Backward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {
  const Dtype* inputs_data = (*inputs)[0]->data();
  const Dtype* inputs_label = (*inputs)[1]->data();
  const Dtype* outputs_data = outputs[0]->data();
  Dtype* inputs_diff = (*inputs)[0]->mutable_data();
  int num = (*inputs)[0]->shape().num();
  int count = (*inputs)[0]->shape().count();
  mll_layer_backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, cuda_stream>>>(num, count/num, kLOG_THRESHOLD, inputs_data,
    inputs_label, outputs_data, inputs_diff);
}

#endif

}  // namespace caffe
