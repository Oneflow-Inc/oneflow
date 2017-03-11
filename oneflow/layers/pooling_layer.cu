#include <algorithm>
#include <string>
#include <vector>

#include "common/common.h"
#include "layers/pooling_layer.h"
#include "math/math_util.h"

namespace caffe {
template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* inputs_data,
  const int num, const int channels, const int height,
  const int width, const int pooled_height, const int pooled_width,
  const int kernel_h, const int kernel_w, const int stride_h,
  const int stride_w, const int pad_h, const int pad_w, Dtype* outputs_data,
  Dtype* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const inputs_slice =
      inputs_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (inputs_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = inputs_slice[maxidx];
        }
      }
    }
    outputs_data[index] = maxval;
    mask[index] = maxidx;
  }
}
template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* inputs_data,
  const int num, const int channels, const int height,
  const int width, const int pooled_height, const int pooled_width,
  const int kernel_h, const int kernel_w, const int stride_h,
  const int stride_w, const int pad_h, const int pad_w, Dtype* outputs_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const inputs_slice =
      inputs_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += inputs_slice[h * width + w];
      }
    }
    outputs_data[index] = aveval / pool_size;
  }
}
template <typename Dtype>
void PoolingLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(PoolingData, data, data_param);
  GET_CONCRETE_POINTER(PoolingParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  const Dtype* inputs_data = data->in->data();
  Dtype* outputs_data = data->out->mutable_data();
  int count = data->out->shape().count();
  switch (param->pool_) {
  case PoolingProto_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, ctx.cuda_stream >> >(count, inputs_data, data->in->shape().num(),
      param->channels_, param->height_, param->width_, param->pooled_height_,
      param->pooled_width_, param->kernel_h_, param->kernel_w_,
      param->stride_h_, param->stride_w_, param->pad_h_, param->pad_w_,
      outputs_data, data->idx->mutable_data());
    break;
  case PoolingProto_PoolMethod_AVE:
    CHECK_NOTNULL(data->idx);
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, ctx.cuda_stream >> >(count, inputs_data, data->in->shape().num(),
      param->channels_, param->height_, param->width_, param->pooled_height_,
      param->pooled_width_, param->kernel_h_, param->kernel_w_,
      param->stride_h_, param->stride_w_, param->pad_h_, param->pad_w_,
      outputs_data);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* outputs_diff,
  const Dtype* mask, const int num, const int channels,
  const int height, const int width, const int pooled_height,
  const int pooled_width, const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w, const int pad_h, const int pad_w,
  Dtype* inputs_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
      (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
      (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const outputs_slice = outputs_diff + offset;
    const Dtype* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += outputs_slice[ph * pooled_width + pw];
        }
      }
    }
    inputs_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* outputs_diff,
  const int num, const int channels, const int height,
  const int width, const int pooled_height, const int pooled_width,
  const int kernel_h, const int kernel_w, const int stride_h,
  const int stride_w, const int pad_h, const int pad_w,
  Dtype* inputs_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const outputs_slice =
      outputs_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += outputs_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    inputs_diff[index] = gradient;
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(PoolingData, data, data_param);
  GET_CONCRETE_POINTER(PoolingParam, param, param_);

  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  // Use ctx, data and model
  CHECK_NOTNULL(data->in);
  CHECK_NOTNULL(data->out);
  CHECK_NOTNULL(data->in_diff);
  CHECK_NOTNULL(data->out_diff);
  const Dtype* outputs_diff_ = data->out_diff->data();
  Dtype* inputs_diff_ = data->in_diff->mutable_data();
  const int count = data->in->shape().count();
  caffe_gpu_async_set(count, Dtype(0.), inputs_diff_, ctx.cuda_stream);
  switch (param->pool_) {
  case PoolingProto_PoolMethod_MAX:
    CHECK_NOTNULL(data->idx);
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, ctx.cuda_stream >> >(count, outputs_diff_, data->idx->data(),
      data->out->shape().num(), param->channels_, param->height_, param->width_,
      param->pooled_height_, param->pooled_width_, param->kernel_h_,
      param->kernel_w_, param->stride_h_, param->stride_w_, param->pad_h_,
      param->pad_w_, inputs_diff_);
    break;
  case PoolingProto_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, ctx.cuda_stream >> >(count, outputs_diff_, data->out->shape().num(),
      param->channels_, param->height_, param->width_, param->pooled_height_,
      param->pooled_width_, param->kernel_h_, param->kernel_w_,
      param->stride_h_, param->stride_w_, param->pad_h_, param->pad_w_,
      inputs_diff_);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}
INSTANTIATE_LAYER_FUNCS(PoolingLayer);
#if 0
template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* inputs_data,
  const int num, const int channels, const int height,
  const int width, const int pooled_height, const int pooled_width,
  const int kernel_h, const int kernel_w, const int stride_h,
  const int stride_w, const int pad_h, const int pad_w, Dtype* outputs_data,
  Dtype* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const inputs_slice =
      inputs_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (inputs_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = inputs_slice[maxidx];
        }
      }
    }
    outputs_data[index] = maxval;
    //if (mask) {
    mask[index] = maxidx;
    //} else {
    //  outputs_mask[index] = maxidx;
    //}
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* inputs_data,
  const int num, const int channels, const int height,
  const int width, const int pooled_height, const int pooled_width,
  const int kernel_h, const int kernel_w, const int stride_h,
  const int stride_w, const int pad_h, const int pad_w, Dtype* outputs_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const inputs_slice =
      inputs_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += inputs_slice[h * width + w];
      }
    }
    outputs_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Forward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {
  const Dtype* inputs_data = inputs[0]->data();
  Dtype* outputs_data = (*outputs)[0]->mutable_data();
  int count = (*outputs)[0]->shape().count();
  // We'll output the mask to outputs[1] if it's of size >1.
  //const bool use_outputs_mask = (*outputs).size() > 1;
  Dtype* mask = nullptr;
  //Dtype* outputs_mask = nullptr;
  switch (pooling_param_.pool()) {
  case PoolingParameter_PoolMethod_MAX:
    //if (use_outputs_mask) {
    //outputs_mask = (*outputs)[1]->mutable_data();
    //} else {
    CHECK(model->find("idx") != model->end());
    mask = model->find("idx")->second->mutable_data();
    //}
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, cuda_stream >> >(count, inputs_data, inputs[0]->shape().num(),
      channels_, height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, outputs_data, mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, cuda_stream >> >(count, inputs_data, inputs[0]->shape().num(),
      channels_, height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, outputs_data);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* outputs_diff,
  const Dtype* mask, const int num, const int channels,
  const int height, const int width, const int pooled_height,
  const int pooled_width, const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w, const int pad_h, const int pad_w,
  Dtype* inputs_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
      (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
      (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const outputs_slice = outputs_diff + offset;
    const Dtype* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += outputs_slice[ph * pooled_width + pw];
        }
      }
    }
    //}
    /*else {
    outputs_mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
    for (int pw = pwstart; pw < pwend; ++pw) {
    if (outputs_mask[ph * pooled_width + pw] == h * width + w) {
    gradient += outputs_diff[ph * pooled_width + pw];
    }
    }
    }
    }*/
    inputs_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* outputs_diff,
  const int num, const int channels, const int height,
  const int width, const int pooled_height, const int pooled_width,
  const int kernel_h, const int kernel_w, const int stride_h,
  const int stride_w, const int pad_h, const int pad_w,
  Dtype* inputs_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const outputs_slice =
      outputs_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += outputs_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    inputs_diff[index] = gradient;
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {
  const Dtype* outputs_diff_ = outputs[0]->data();
  Dtype* inputs_diff_ = (*inputs)[0]->mutable_data();
  const int count = (*inputs)[0]->shape().count();
  caffe_gpu_set(count, Dtype(0.), inputs_diff_, cuda_stream);
  // We'll output the mask to outputs[1] if it's of size >1.
  //const bool use_outputs_mask_ = outputs.size() > 1;
  const Dtype* mask_ = nullptr;
  //const Dtype* outputs_mask_ = nullptr;
  switch (pooling_param_.pool()) {
  case PoolingParameter_PoolMethod_MAX:
    //if (use_outputs_mask_) {
    //outputs_mask_ = outputs[1]->data();
    //} else {
    CHECK(model->find("idx") != model->end());
    mask_ = model->find("idx")->second->mutable_data();
    //}
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, cuda_stream >> >(count, outputs_diff_, mask_,
      outputs[0]->shape().num(), channels_, height_, width_, pooled_height_,
      pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
      inputs_diff_);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
      0, cuda_stream >> >(count, outputs_diff_, outputs[0]->shape().num(),
      channels_, height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, inputs_diff_);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}
#endif
}  // namespace caffe
