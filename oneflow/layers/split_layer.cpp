#include <cstdint>
#include <vector>
#include <string>
#include "layers/split_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "math\math_util.h"

namespace caffe {
template <typename Dtype>
void SplitLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new SplitParam<Dtype>();
  SplitProto split_proto;
  ParseProtoFromStringOrDie(proto_param_, &split_proto);
  // Set the number of output blobs
  CHECK_GT(split_proto.out_num(), 1);
  param->out_num_ = split_proto.out_num();
  param_ = param;
}
template <typename Dtype>
void SplitLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(SplitData, data, data_param);
  GET_CONCRETE_POINTER(SplitParam, param, param_);

  // Set the shape of output blobs
  auto& in_shape = data->in->shape();
  for (int32_t idx = 0; idx < param->out_num_; ++idx) {
    data->out[idx]->set_shape(in_shape);
  }

  // FIXME(jiyuan): set the diff blob shape
  // Copy the blob shapes in data_param to this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}
template <typename Dtype>
void SplitLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(SplitData, data, data_param);
  GET_CONCRETE_POINTER(SplitModel, model, model_param);
  // Use ctx, data and model
  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  const Shape& in_shape = data->in->shape();
  for (int i = 0; i < data->out.size(); ++i) {
    CUDA_CHECK(cudaMemcpyAsync(data->out[i]->mutable_data(),
      data->in->data(), in_shape.count() * sizeof(Dtype),
      cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  }
}
template <typename Dtype>
void SplitLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(SplitData, data, data_param);
  GET_CONCRETE_POINTER(SplitModel, model, model_param);
  // Use ctx, data and model
  CHECK(ctx.cuda_stream) << "Default stream is not allowed";

  const Shape& in_shape = data->in->shape();
  if (data->out.size() == 1) {
    CUDA_CHECK(cudaMemcpyAsync(data->in->mutable_data(),
      data->out[0]->data(), in_shape.count() * sizeof(Dtype),
      cudaMemcpyDeviceToDevice, ctx.cuda_stream));
  } else {
    Dtype* in_diff = data->in->mutable_data();
    caffe_gpu_add(in_shape.count(), data->out[0]->data(), 
      data->out[1]->data(), in_diff, ctx.cuda_stream);
    // Add remaining top blob diffs.
    for (int i = 2; i < data->out.size(); ++i) {
      const Dtype* out_diff = data->out[i]->data();
      caffe_gpu_axpy(ctx.cublas_handle, in_shape.count(), Dtype(1.), 
        out_diff, in_diff, ctx.cuda_stream);
    }
  }
}
INSTANTIATE_LAYER_FUNCS(SplitLayer);
INSTANTIATE_CLASS(SplitLayer);
REGISTER_LAYER_CLASS(Split);
}  // namespace caffe
