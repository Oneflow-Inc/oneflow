#include <cstdint>
#include <string>
#include <vector>

#include "layers/data_layers.h"
#include "common/common.h"
#include "common/shape.h"
#include "layers/layer_factory.h"

namespace caffe {
  template <typename Dtype>
  void DataCopyLayer<Dtype>::LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs) {
    DLOG(INFO) << "Settingup DataCopyLayer...";
  }
  template <typename Dtype>
  void DataCopyLayer<Dtype>::Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs) {
    CHECK_EQ(1, inputs.size()) << "Only support one output";
    CHECK_EQ(1, outputs->size()) << "Only support one output";
    const Shape& in_shape = inputs[0]->immutable_shape();
    Shape& out_shape = (*outputs)[0]->mutable_shape();
    out_shape.Reshape(in_shape.immutable_shape());
  }

  INSTANTIATE_CLASS(DataCopyLayer);
  REGISTER_LAYER_CLASS(DataCopy);
}  // namespace caffe
