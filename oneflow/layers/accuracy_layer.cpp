#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "common/common.h"
#include "layers/layer.h"
#include "layers/layer_factory.h"
#include "layers/loss_layers.h"
#include "proto_io.h"
#include "math/math_util.h"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetup(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  top_k_ = accuracy_param_.top_k();

  has_ignore_label_ = accuracy_param_.has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = accuracy_param_.ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  CHECK_LE(top_k_, inputs[0]->immutable_shape().count() /
    inputs[1]->immutable_shape().count())
    << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
    inputs[0]->immutable_shape().CanonicalAxisIndex(accuracy_param_.axis());
  outer_num_ = inputs[0]->immutable_shape().count(0, label_axis_);
  inner_num_ = inputs[0]->immutable_shape().count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, inputs[1]->immutable_shape().count())
    << "Number of labels must match number of predictions; "
    << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
    << "label count (number of labels) must be N*H*W, "
    << "with integer values in {0, 1, ..., C-1}.";
  std::vector<int64_t> outputs_shape(0);  // Accuracy is a scalar; 0 axes.
  (*outputs)[0]->mutable_shape().Reshape(outputs_shape);
  if (outputs->size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    std::vector<int64_t> outputs_shape_per_class(1);
    outputs_shape_per_class[0] =
      inputs[0]->immutable_shape().shape(label_axis_);
    (*outputs)[1]->mutable_shape().Reshape(outputs_shape_per_class);
    ParameterReshape("nums_buffer", outputs_shape_per_class);
  }
}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
