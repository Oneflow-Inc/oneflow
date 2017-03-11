#include <cstdint>
#include <vector>
#include <string>
#include "layers/innerproduct_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void InnerProductLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new InnerProductParam<Dtype>();

  InnerProductProto innerproduct_proto;
  ParseProtoFromStringOrDie(proto_param_, &innerproduct_proto);
  param->axis_ = innerproduct_proto.axis();
  param->num_output_ = innerproduct_proto.num_output();;
  param->bias_term_ = innerproduct_proto.bias_term();

  param_ = param;
}
template <typename Dtype>
void InnerProductLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(InnerProductData, data, data_param);
  GET_CONCRETE_POINTER(InnerProductParam, param, param_);
  auto model_param = param->mutable_model_param();
  GET_CONCRETE_POINTER(InnerProductModel, model, model_param);

  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  const Shape& in_shape = data->in->shape();
  data->in_diff->set_shape(in_shape);

  const int32_t axis = in_shape.CanonicalAxisIndex(param->axis_);
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  param->num_example_ = in_shape.count(0, axis);
  param->num_input_ = in_shape.count(axis);

  std::vector<int64_t> weight_shape;
  weight_shape.assign({ param->num_output_, param->num_input_ });
  model->weight->set_shape(weight_shape);
  model->weight_diff->set_shape(weight_shape);
  if (param->bias_term_) {
    std::vector<int64_t> bias_shape(1, param->num_output_);
    model->bias->set_shape(bias_shape);
    model->bias_diff->set_shape(bias_shape);
    std::vector<int64_t> bias_multiplier_shape(1, param->num_example_);
    model->bias_multiplier->set_shape(bias_multiplier_shape);
  }

  std::vector<int64_t> output_shape;
  output_shape.assign({ in_shape.shape(0), param->num_output_ });
  data->out->set_shape(output_shape);
  data->out_diff->set_shape(output_shape);

  // NOTE(jiyuan): remember to align the shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);
}  // namespace caffe
