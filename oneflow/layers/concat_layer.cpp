#include <cstdint>
#include <vector>
#include <string>
#include "layers/concat_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void ConcatLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new ConcatParam<Dtype>();
  // Get the number of inputs from proto
  ConcatProto concat_proto;
  ParseProtoFromStringOrDie(proto_param_, &concat_proto);
  CHECK(concat_proto.has_in_num());
  param->in_num_ = concat_proto.in_num();
  CHECK(concat_proto.has_axis());
  param->axis_ = concat_proto.axis();
  param_ = param;
}
template <typename Dtype>
void ConcatLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(ConcatData, data, data_param);
  GET_CONCRETE_POINTER(ConcatParam, param, param_);

  CHECK(data->in.size() >= 1);
  const Shape& in_shape = data->in[0]->shape();
  Shape out_shape = in_shape;
  const int32_t num_axes = in_shape.num_axes();
  const int32_t axis = in_shape.CanonicalAxisIndex(param->axis_);
  param->num_concats_ = in_shape.count(0, axis);
  param->concat_input_size_ = in_shape.count(axis + 1);
  int64_t bottom_count_sum = in_shape.count();
  int64_t concat_axis_dim = in_shape.shape(axis);
  for (int32_t i = 1; i < data->in.size(); ++i) {
    const Shape& cur_shape = data->in[i]->shape();
    CHECK_EQ(num_axes, cur_shape.num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == axis) { continue; }
      CHECK_EQ(in_shape.shape(j), cur_shape.shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    bottom_count_sum += cur_shape.count();
    concat_axis_dim += cur_shape.shape(axis);
  }
  out_shape.set_shape(axis, concat_axis_dim);
  data->out->mutable_shape() = out_shape;
  CHECK_EQ(bottom_count_sum, out_shape.count());

  // TODO(jiyuan): verify whether it needs sharing input & output
  //top[0]->Reshape(top_shape);
  //if (bottom.size() == 1) {
  //  top[0]->ShareData(*bottom[0]);
  //  top[0]->ShareDiff(*bottom[0]);
  //}
  // FIXME(jiyuan): set the shape of diff blob

  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}

INSTANTIATE_CLASS(ConcatLayer);
REGISTER_LAYER_CLASS(Concat);
}  // namespace caffe
