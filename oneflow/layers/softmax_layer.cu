#include <algorithm>
#include <string>
#include <vector>

#include "common/common.h"
#include "layers/softmax_layer.h"
#include "math/math_util.h"

namespace caffe {
template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
  const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
  const int num, const int channels,
  const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
  const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
  const int num, const int channels,
  const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
  const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
  Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
        * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(SoftmaxData, data, data_param);
  GET_CONCRETE_POINTER(SoftmaxParam, param, param_);
  GET_CONCRETE_POINTER(SoftmaxModel, model, model_param);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(model->scale);

  const Dtype* inputs_data = data->in->data();
  Dtype* outputs_data = data->out->mutable_data();
  Dtype* scale_data = model->scale->mutable_data();
  int count = data->in->shape().count();
  int channels = data->out->shape().shape(param->softmax_axis_);
  caffe_gpu_async_copy(count, inputs_data, outputs_data, ctx.cuda_stream);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  kernel_channel_max<Dtype> << <CAFFE_GET_BLOCKS(
    param->outer_num_ * param->inner_num_), CAFFE_CUDA_NUM_THREADS, 0,
    ctx.cuda_stream >> >(param->outer_num_, channels, param->inner_num_,
    outputs_data, scale_data);
  // subtract
  kernel_channel_subtract<Dtype> << <CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS, 0, ctx.cuda_stream >> >(count, param->outer_num_,
    channels, param->inner_num_, scale_data, outputs_data);
  // exponentiate
  kernel_exp<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
    ctx.cuda_stream >> >(count, outputs_data, outputs_data);
  // sum after exp
  kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(
    param->outer_num_ * param->inner_num_), CAFFE_CUDA_NUM_THREADS, 0,
    ctx.cuda_stream >> >(param->outer_num_, channels, param->inner_num_,
    outputs_data, scale_data);
  // divide
  kernel_channel_div<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream >> >(count, param->outer_num_, channels,
    param->inner_num_, scale_data, outputs_data);
}
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(SoftmaxData, data, data_param);
  GET_CONCRETE_POINTER(SoftmaxParam, param, param_);
  GET_CONCRETE_POINTER(SoftmaxModel, model, model_param);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  // Use ctx, data and model
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);

  CHECK_NOTNULL(model->scale);
  const Dtype* outputs_data = data->out->data();
  Dtype* outputs_diff = data->out_diff->mutable_data();
  Dtype* inputs_diff = data->in_diff->mutable_data();
  Dtype* scale_data = model->scale->mutable_data();
  int count = data->out->shape().count();
  int channels = data->out->shape().shape(param->softmax_axis_);
  caffe_gpu_async_copy(count, outputs_data, inputs_diff, ctx.cuda_stream);
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(
    param->outer_num_ * param->inner_num_), CAFFE_CUDA_NUM_THREADS, 0, 
    ctx.cuda_stream>>>(param->outer_num_, channels, param->inner_num_,
    outputs_diff, inputs_diff, scale_data);
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS, 0, ctx.cuda_stream>>>(count, param->outer_num_,
    channels, param->inner_num_, scale_data, outputs_diff);
  // element-wise multiplication
  caffe_gpu_mul<Dtype>(data->out->shape().count(), outputs_diff, inputs_diff,
    inputs_diff, ctx.cuda_stream);
}
INSTANTIATE_LAYER_FUNCS(SoftmaxLayer);
#if 0
template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
  const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
  const int num, const int channels,
  const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
  const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
  const int num, const int channels,
  const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
  const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
  Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
        * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {
  CHECK(model->find("scale") != model->end());
  Dtype* inputs_data = inputs[0]->mutable_data();
  Dtype* outputs_data = (*outputs)[0]->mutable_data();
  Dtype* scale_data = model->find("scale")->second->mutable_data();
  int count = inputs[0]->shape().count();
  int channels = (*outputs)[0]->shape().shape(softmax_axis_);
  caffe_copy(count, inputs_data, outputs_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
    CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>(outer_num_, channels, inner_num_,
    outputs_data, scale_data);
  // subtract
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>(count, outer_num_, channels,
    inner_num_, scale_data, outputs_data);
  // exponentiate
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
    cuda_stream>>>(count, outputs_data, outputs_data);
  // sum after exp
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
    CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>(outer_num_, channels, inner_num_,
    outputs_data, scale_data);
  // divide
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, cuda_stream>> >(count, outer_num_, channels, inner_num_, scale_data,
    outputs_data);
  caffe_copy(count, outputs_data, inputs_data);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {
  CHECK(model->find("scale") != model->end());
  Dtype* outputs_diff = outputs[0]->mutable_data();
  Dtype* inputs_data = (*inputs)[0]->mutable_data();
  Dtype* scale_data = model->find("scale")->second->mutable_data();
  int count = outputs[0]->shape().count();
  int channels = outputs[0]->shape().shape(softmax_axis_);
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
    CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>(outer_num_, channels, inner_num_,
    outputs_diff, inputs_data, scale_data);
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>(count, outer_num_, channels,
    inner_num_, scale_data, outputs_diff);
  // element-wise multiplication
  caffe_gpu_mul<Dtype>(outputs[0]->shape().count(), outputs_diff, inputs_data,
    inputs_data, cuda_stream);
}
#endif


}  // namespace caffe
