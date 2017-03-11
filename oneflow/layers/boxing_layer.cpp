#include <cstdint>
#include <vector>
#include <string>
#include "layers/boxing_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "math/math_util.h"
#include "common/split_util.h"

namespace caffe {
template <typename Dtype>
void BoxingLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new BoxingParam<Dtype>();
  BoxingProto boxing_proto;
  ParseProtoFromStringOrDie(proto_param_, &boxing_proto);

  CHECK(boxing_proto.has_in_num());
  param->in_num_ = boxing_proto.in_num();
  CHECK(boxing_proto.has_in_op());
  param->in_op_ = boxing_proto.in_op();
  CHECK(boxing_proto.has_backward_in_op());
  param->backward_in_op_ = boxing_proto.backward_in_op();
  CHECK(boxing_proto.has_in_axis());
  param->in_axis_ = boxing_proto.in_axis();

  CHECK(boxing_proto.has_out_num());
  param->out_num_ = boxing_proto.out_num();
  CHECK(boxing_proto.has_out_op());
  param->out_op_ = boxing_proto.out_op();
  CHECK(boxing_proto.has_backward_out_op());
  param->backward_out_op_ = boxing_proto.backward_out_op();
  CHECK(boxing_proto.has_out_axis());
  param->out_axis_ = boxing_proto.out_axis();

  param_ = param;
}
template <typename Dtype>
void BoxingLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(BoxingData, data, data_param);
  GET_CONCRETE_POINTER(BoxingParam, param, param_);
  Shape middle_shape;
  CHECK(param->in_num_ >= 1);
  CHECK(param->out_num_ >= 1);
  if (param->in_num_ == 1) {
    middle_shape = data->in[0]->shape();
  } else {
    const Shape& in_shape = data->in[0]->shape();
    middle_shape = in_shape;
    // For ADD, ensure every input blob has the same shape
    if (param->in_op_ == ADD) {
      for (int i = 1; i < param->in_num_; i++) {
        CHECK(middle_shape == data->in[i]->shape());
      }
    } else {
      // CONCAT, axis to infer middle_shape
      const int32_t num_axes = in_shape.num_axes();
      const int32_t axis = in_shape.CanonicalAxisIndex(param->in_axis_);
      int64_t num_concats = in_shape.count(0, axis);
      int64_t concat_input_size = in_shape.count(axis + 1);
      int64_t bottom_count_sum = in_shape.count();
      int64_t concat_axis_dim = in_shape.shape(axis);
      for (int32_t i = 1; i < param->in_num_; ++i) {
        const Shape& cur_shape = data->in[i]->shape();
        CHECK_EQ(num_axes, cur_shape.num_axes())
          << "All inputs must have the same #axes";
        for (int j = 0; j < num_axes; ++j) {
          if (j == axis) continue;
          CHECK_EQ(in_shape.shape(j), cur_shape.shape(j))
            << "All inputs must have the same shape, except at concat_axis";
        }
        bottom_count_sum += cur_shape.count();
        concat_axis_dim += cur_shape.shape(axis);
      }
      middle_shape.set_shape(axis, concat_axis_dim);
      CHECK_EQ(middle_shape.count(), bottom_count_sum);
    }
  }
  // Set the middle blob's shape
  data->middle->mutable_shape() = middle_shape;
  if (param->out_op_ == COPY) {
    // All the output blobs have the same shape
    for (int32_t idx = 0; idx < param->out_num_; ++idx) {
      data->out[idx]->mutable_shape() = middle_shape;
    }
  } else if (param->out_op_ == SPLIT) {
    const int32_t axis = middle_shape.CanonicalAxisIndex(param->out_axis_);
    int64_t split_axis_dim = middle_shape.shape(axis);
    std::vector<int64_t> split_dims;
    GetDimOfEachSplit(split_axis_dim, param->out_num_, &split_dims);
    for (int32_t idx = 0; idx < param->out_num_; ++idx) {
      data->out[idx]->mutable_shape() = middle_shape;
      data->out[idx]->mutable_shape().set_shape(axis, split_dims[idx]);
    }
  } else {
    LOG(FATAL) << "Impossible out op type";
  }
  // Set the blobs' shape in param_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
  // TODO(jiyuan): set the diff blob shape
}
template <typename Dtype>
void BoxingLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(BoxingData, data, data_param);
  GET_CONCRETE_POINTER(BoxingModel, model, model_param);
  GET_CONCRETE_POINTER(BoxingParam, param, param_);
  // Use ctx, data and model

  // TODO(jiyuan): can we implement boxing without using the middle blob?

  switch (param->in_op_) {
  case CONCAT: {
    Dtype* middle_data = data->middle->mutable_data();
    int offset_concat_axis = 0;
    const int middle_concat_axis = data->middle->shape().shape(param->in_axis_);
    int64_t num_concats = data->in[0]->shape().count(0, param->in_axis_);
    int64_t concat_input_size = data->in[0]->shape().count(param->in_axis_ + 1);
    for (int i = 0; i < data->in.size(); ++i) {
      const Dtype* in_data = data->in[i]->data();
      const int in_concat_axis = data->in[i]->shape().shape(param->in_axis_);
      for (int n = 0; n < num_concats; ++n) {
        caffe_copy(in_concat_axis * concat_input_size,
          in_data + n * in_concat_axis * concat_input_size,
          middle_data + (n * middle_concat_axis + offset_concat_axis)
          * concat_input_size);
      }
      offset_concat_axis += in_concat_axis;
    }
  }
    break;
  case ADD: {
    Dtype* middle_data = data->middle->mutable_data();
    auto& first_in_shape = data->in[0]->shape();
    caffe_copy(first_in_shape.count(), data->in[0]->data(), middle_data);
    for (int i = 1; i < data->in.size(); i++) {
      caffe_add<Dtype>(first_in_shape.count(), middle_data,
        data->in[i]->data(), middle_data);
    }
  }
  default:
    LOG(FATAL) << "Unsupported in_op for boxing layer in Forward";
    break;
  }

  switch (param->out_op_) {
  case COPY: {
    const Dtype* middle_data = data->middle->data();
    const int32_t count = data->middle->shape().count();
    for (int32_t idx = 0; idx < param->out_num_; ++idx) {
      caffe_copy(count, middle_data, data->out[idx]->mutable_data());
    }
  }
    break;
  case SPLIT: {
    const Dtype* middle_data = data->middle->data();
    int offset_split_axis = 0;
    const int middle_split_axis = data->middle->shape().shape(param->out_axis_);
    int64_t num_splits = data->middle->shape().count(0, param->out_axis_);
    int64_t split_input_size = data->middle->shape().count(param->out_axis_ + 1);
    for (int i = 0; i < data->out.size(); ++i) {
      Dtype* out_data = data->out[i]->mutable_data();
      const int out_split_axis = data->out[i]->shape().shape(param->out_axis_);
      for (int n = 0; n < num_splits; ++n) {
        caffe_copy(out_split_axis * split_input_size,
          middle_data + (n * middle_split_axis + offset_split_axis)
          * split_input_size,
          out_data + n * out_split_axis * split_input_size);
      }
      offset_split_axis += out_split_axis;
    }
  }
    break;
  default:
    LOG(FATAL) << "Unsupported out op for boxing layer in Forward";
    break;
  }
}
template <typename Dtype>
void BoxingLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(BoxingData, data, data_param);
  GET_CONCRETE_POINTER(BoxingModel, model, model_param);
  GET_CONCRETE_POINTER(BoxingParam, param, param_);
  // Each boxing_layer serves a particular logical_blob, therefore, the 
  // channel_is_enable values for all the channels should be with the same value,
  // either all false, or all true.
  bool enable = data->channel_is_enabled(0);
  for (int i = 1; i < data->out.size(); ++i) {
    CHECK_EQ(enable, data->channel_is_enabled(i));
  }
  if (!enable) return;

  // Use ctx, data and model
  switch (param->out_op_) {
  case COPY: {
      CHECK(param->backward_out_op_ == ADD) << "COPY<->ADD";
      const int count = data->middle->shape().count();
      Dtype* middle = data->middle->mutable_data();
      caffe_add(count, data->out[0]->data(), data->out[1]->data(),
        middle);
      // Add remaining top blob diffs.
      for (int i = 2; i < data->out.size(); ++i) {
        const Dtype* out = data->out[i]->data();
        caffe_axpy(count, Dtype(1.), out, middle);
      }
  }
    break;
  case SPLIT: {
    CHECK(param->backward_out_op_ == CONCAT) << "SPLIT<->CONCAT";
    Dtype* middle = data->middle->mutable_data();
    int offset_split_axis = 0;
    const int middle_split_axis = data->middle->shape().shape(param->out_axis_);
    int64_t num_splits = data->middle->shape().count(0, param->out_axis_);
    int64_t split_input_size = data->middle->shape().count(param->out_axis_ + 1);
    for (int i = 0; i < data->out.size(); ++i) {
      const Dtype* out = data->out[i]->data();
      const int out_split_axis = data->out[i]->shape().shape(param->out_axis_);
      for (int n = 0; n < num_splits; ++n) {
        caffe_copy(out_split_axis * split_input_size,
          out + n * out_split_axis * split_input_size,
          middle + (n * middle_split_axis + offset_split_axis)
          * split_input_size);
      }
      offset_split_axis += out_split_axis;
    }
  }
    break;
  default:
    LOG(FATAL) << "Unsupported out op for boxing layer in Backward";
    break;
  }

  switch (param->in_op_) {
  case CONCAT: {
    CHECK(param->backward_in_op_ == SPLIT) << "";
    const Dtype* middle = data->middle->data();
    int offset_concat_axis = 0;
    const int middle_concat_axis = data->middle->shape().shape(param->in_axis_);
    int64_t num_concats = data->middle->shape().count(0, param->in_axis_);
    int64_t concat_input_size = data->middle->shape().count(param->in_axis_ + 1);
    for (int i = 0; i < data->in.size(); ++i) {
      Dtype* in = data->in[i]->mutable_data();
      const int in_concat_axis = data->in[i]->shape().shape(param->in_axis_);
      for (int n = 0; n < num_concats; ++n) {
        caffe_copy(in_concat_axis * concat_input_size,
          middle + (n * middle_concat_axis + offset_concat_axis)
          * concat_input_size,
          in + n * in_concat_axis * concat_input_size);
      }
      offset_concat_axis += in_concat_axis;
    }
  }
    break;
  default:
    LOG(FATAL) << "Unsupported out op for boxing layer in Backward";
    break;
  }
}
INSTANTIATE_LAYER_FUNCS(BoxingLayer);
INSTANTIATE_CLASS(BoxingLayer);
REGISTER_LAYER_CLASS(Boxing);
}  // namespace caffe
