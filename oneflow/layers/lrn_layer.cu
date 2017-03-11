#include <algorithm>
#include <string>
#include <vector>

#include "common/common.h"
#include "layers/lrn_layer.h"
#include "math/math_util.h"

namespace caffe {

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* const in,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype alpha_over_size,
  const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
          * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
          * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* const in,
  const Dtype* const scale, const Dtype negative_beta, Dtype* const out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(LRNData, data, data_param);
  GET_CONCRETE_POINTER(LRNParam, param, param_);

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);

  // First, compute scale
  const Dtype* in_data = data->in->data();
  Dtype* out_data = data->out->mutable_data();
  Dtype* scale_data = data->scale->mutable_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = param->num_ * param->height_ * param->width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScale << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream >> >(n_threads, in_data, param->num_, param->channels_,
    param->height_, param->width_, param->size_, param->alpha_ / param->size_,
    param->k_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = data->in->shape().count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream >> >(n_threads, in_data, scale_data,
    -(param->beta_), out_data);
  CUDA_POST_KERNEL_CHECK;
}
template void LRNLayer<float>::CrossChannelForward(const ContextParam& ctx,
  DataParam<float>* data_param, ModelParam<float>* model_param) const;
template void LRNLayer<double>::CrossChannelForward(const ContextParam& ctx,
  DataParam<double>* data_param, ModelParam<double>* model_param) const;

template <typename Dtype>
void LRNLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(LRNData, data, data_param);
  GET_CONCRETE_POINTER(LRNParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  // CHECK_NOTNULL(data->in_copy);
  // CHECK_NOTNULL(data->out_copy);


  switch (param->norm_region_) {
  case LRNProto_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward(ctx, data_param, model_param);
    break;
  case LRNProto_NormRegion_WITHIN_CHANNEL:
    // To do ...
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }

  // CUDA_CHECK(cudaMemcpyAsync(data->in_copy->mutable_data(),
  //   data->in->data(), data->in->shape().count()*sizeof(Dtype),
  //   cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  // CUDA_CHECK(cudaMemcpyAsync(data->out_copy->mutable_data(),
  //   data->out->data(), data->out->shape().count()*sizeof(Dtype),
  //   cudaMemcpyDeviceToDevice, ctx.cuda_stream));



}


template <typename Dtype>
__global__ void LRNComputeDiff(const int nthreads,
  const Dtype* const bottom_data, const Dtype* const top_data,
  const Dtype* const scale, const Dtype* const top_diff,
  const int num, const int channels, const int height,
  const int width, const int size, const Dtype negative_beta,
  const Dtype cache_ratio, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
        scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
        scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
          top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
        top_diff_off[(head - post_pad) * step]
        * pow(scale_off[(head - post_pad) * step], negative_beta)
        - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
          top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
        top_diff_off[(head - post_pad) * step]
        * pow(scale_off[(head - post_pad) * step], negative_beta)
        - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {

  GET_CONCRETE_POINTER(LRNData, data, data_param);
  GET_CONCRETE_POINTER(LRNParam, param, param_);

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);


  int n_threads = param->num_ * param->height_ * param->width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS,
    0, ctx.cuda_stream >> >(
    n_threads, data->in->data(), data->out->data(),
    data->scale->data(), data->out_diff->data(), param->num_, param->channels_,
    param->height_, param->width_,
    param->size_, -(param->beta_),
    Dtype(2. * param->alpha_ * param->beta_ / param->size_),
    data->in_diff->mutable_data());
}
template void LRNLayer<float>::CrossChannelBackward(const ContextParam& ctx,
  DataParam<float>* data_param, ModelParam<float>* model_param) const;
template void LRNLayer<double>::CrossChannelBackward(const ContextParam& ctx,
  DataParam<double>* data_param, ModelParam<double>* model_param) const;



template <typename Dtype>
void LRNLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(LRNParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  switch (param->norm_region_) {
  case LRNProto_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward(ctx, data_param, model_param);
    break;
  case LRNProto_NormRegion_WITHIN_CHANNEL:
    // To do ...
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}



INSTANTIATE_LAYER_FUNCS(LRNLayer);

}  // namespace caffe
