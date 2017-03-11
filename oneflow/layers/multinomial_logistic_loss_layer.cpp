#include <cstdint>
#include <vector>
#include "layers/multinomial_logistic_loss_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new MultinomialLogisticLossParam<Dtype>();
  MultinomialLogisticLossProto multinomiallogisticloss_proto;
  ParseProtoFromStringOrDie(proto_param_, &multinomiallogisticloss_proto);
  param_ = param;
}
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(MultinomialLogisticLossData, data, data_param);
  GET_CONCRETE_POINTER(MultinomialLogisticLossParam, param, param_);
  auto model_param = param->mutable_model_param();
  GET_CONCRETE_POINTER(MultinomialLogisticLossModel, model, model_param);

  CHECK_EQ(data->data->shape().num(),
    data->label->shape().num()) << "The data and label should have "
    << "the same number.";

  // Shape the loss_buffer & loss_multiplier
  const Shape& in_shape = data->data->shape();
  data->data_diff->set_shape(in_shape);
  std::vector<int64_t> input_shape;
  input_shape.assign({ in_shape.num(), in_shape.dim()});
  data->loss_buffer->set_shape(input_shape);
  model->loss_multiplier->set_shape(input_shape);

  //Shape the loss
  Shape& loss_shape = data->loss->mutable_shape();
  std::vector<int64_t> out_shape(0);
  loss_shape.Reshape(out_shape);

  // NOTE(jiyuan): remember to align the shapes in this->param_->prototype_data_
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}
INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MultinomialLogisticLoss);
#if 0
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::LayerSetup(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  DLOG(INFO) << "Setting up MultinomialLogisticLossLayer...";
}

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Reshape(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  CHECK_EQ(2, inputs.size());
  CHECK_EQ(inputs[0]->shape().num(),
    inputs[1]->shape().num()) << "The data and label should have "
    << "the same number.";
  const Shape& in_shape = inputs[0]->shape();
  std::vector<int64_t> input_shape;
  input_shape.assign({ in_shape.num(), in_shape.dim()});
  ParameterReshape("loss_multiplier", input_shape);
  ParameterReshape("loss_buffer", input_shape);
  Shape& out_shape = (*outputs)[0]->mutable_shape();
  std::vector<int64_t> loss_shape(0);
  out_shape.Reshape(loss_shape);
}
#endif

}  // namespace caffe
