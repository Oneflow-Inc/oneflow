#include <algorithm>
#include <string>
#include <vector>

#include "common/common.h"
#include "layers/concat_layer.h"
#include "math/math_util.h"


namespace caffe {

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
  const bool forward, const int num_concats, const int concat_size,
  const int top_concat_axis, const int bottom_concat_axis,
  const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
      (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ConcatData, data, data_param);
  GET_CONCRETE_POINTER(ConcatModel, model, model_param);
  GET_CONCRETE_POINTER(ConcatParam, param, param_);
  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  // TODO:if in size equals to 1, copy directly.  
  // To be optimized.  
  if (data->in.size() == 1) {
    CUDA_CHECK(cudaMemcpyAsync(data->out->mutable_data(),
      data->in[0]->data(), data->in[0]->shape().count() * sizeof(Dtype),
      cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  } else {
    Dtype* out_data = data->out->mutable_data();
    int offset_concat_axis = 0;
    const int out_concat_axis = data->out->shape().shape(param->axis_);
    const bool kForward = true;
    for (int i = 0; i < param->in_num_; ++i) {
      const Dtype* in_data = data->in[i]->data();
      const int in_concat_axis = data->in[i]->shape().shape(param->axis_);
      const int in_concat_size = in_concat_axis * param->concat_input_size_;
      const int nthreads = in_concat_size * param->num_concats_;

      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0, ctx.cuda_stream >> >(
        nthreads, in_data, kForward, param->num_concats_,
        param->concat_input_size_, out_concat_axis, in_concat_axis,
        offset_concat_axis, out_data);
      offset_concat_axis += in_concat_axis;
    }
  }
}
template <typename Dtype>
void ConcatLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(ConcatData, data, data_param);
  GET_CONCRETE_POINTER(ConcatModel, model, model_param);
  GET_CONCRETE_POINTER(ConcatParam, param, param_);
  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  // TODO:if in size equals to 1, copy directly.  
  // To be optimized.
  if (data->in.size() == 1) {
    CUDA_CHECK(cudaMemcpyAsync(data->in[0]->mutable_data(),
      data->out->data(), data->in[0]->shape().count() * sizeof(Dtype),
      cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  } else {
    const Dtype* out_diff = data->out->mutable_data();
    int offset_concat_axis = 0;
    const int out_concat_axis = data->out->shape().shape(param->axis_);
    const bool kForward = false;
    for (int i = 0; i < param->in_num_; ++i) {
      Dtype* in_diff = data->in[i]->mutable_data();
      const int in_concat_axis = data->in[i]->shape().shape(param->axis_);
      const int in_concat_size = in_concat_axis * param->concat_input_size_;
      const int nthreads = in_concat_size * param->num_concats_;
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0, ctx.cuda_stream >> >(
        nthreads, out_diff, kForward, param->num_concats_,
        param->concat_input_size_, out_concat_axis, in_concat_axis,
        offset_concat_axis, in_diff);
      offset_concat_axis += in_concat_axis;
    }
  }
}

INSTANTIATE_LAYER_FUNCS(ConcatLayer);
}