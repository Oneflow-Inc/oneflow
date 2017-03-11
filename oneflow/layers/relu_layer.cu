#include <algorithm>
#include <string>
#include <vector>

#include "common/common.h"
#include "layers/relu_layer.h"

namespace caffe {
template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
  Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}
template <typename Dtype>
void ReLULayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ReLUData, data, data_param);
  GET_CONCRETE_POINTER(ReLUParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  const Dtype* inputs_data = data->in->data();
  Dtype* outputs_data = data->out->mutable_data();
  const int count = data->in->shape().count();
  ReLUForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream >> >(count, inputs_data, outputs_data,
    param->negative_slope_);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
  const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0 ? 1.0 : 0)
      + (in_data[index] <= 0 ? 1.0 : 0) * negative_slope);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ReLUData, data, data_param);
  GET_CONCRETE_POINTER(ReLUParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);

  const Dtype* outputs_diff_ = data->out_diff->data();
  const Dtype* inputs_data_ = data->in->data();
  Dtype* inputs_diff_ = data->in_diff->mutable_data();
  const int count = data->in->shape().count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream>> >(count, outputs_diff_, inputs_data_, inputs_diff_,
    param->negative_slope_);
  CUDA_POST_KERNEL_CHECK;
}
INSTANTIATE_LAYER_FUNCS(ReLULayer);
#if 0
template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
  Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {
  const Dtype* inputs_data = inputs[0]->data();
  Dtype* outputs_data = (*outputs)[0]->mutable_data();
  const int count = inputs[0]->shape().count();
  // Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // TODO(xcdu) :temporary solution.Should finish the function in future.
  Dtype negative_slope = 0;
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, cuda_stream>> >(count, inputs_data, outputs_data, negative_slope);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
  const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0 ? 1.0 : 0)
      + (in_data[index] <= 0 ? 1.0 : 0) * negative_slope);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {
  const Dtype* outputs_diff_ = outputs[0]->data();
  const Dtype* inputs_data_ = (*inputs)[0]->data();
  Dtype* inputs_diff_ = (*inputs)[0]->mutable_data();
  const int count_ = (*inputs)[0]->shape().count();
  // NOTE(xcdu) :temporary assuming that nagative_slope = 0;
  Dtype negative_slope_ = 0;
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS,
    0, cuda_stream>> >(count_, outputs_diff_, inputs_data_, inputs_diff_,
    negative_slope_);
  CUDA_POST_KERNEL_CHECK;
}
#endif

}  // namespace caffe
