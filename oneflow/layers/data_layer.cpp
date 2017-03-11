#include <cstdint>
#include <vector>
#include "layers/data_layers.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
template <typename Dtype>
void DataLayer<Dtype>::LayerSetup(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  DLOG(INFO) << "Settingup DataLayer...";
}
template <typename Dtype>
void DataLayer<Dtype>::Reshape(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  CHECK_EQ(inputs.size(), 0);   // no other denpendencies
  CHECK_EQ(outputs->size(), 2);  // data and label
  int64_t batch_size = data_param_.batch_size();
  CHECK(data_param_.has_transform_param());
  int64_t channels = data_param_.transform_param().channel();
  int64_t crop_size = data_param_.transform_param().crop_size();

  // NOTE(jiyuan): the output blobs must be named with 'data' and 'label'
  for (int32_t id = 0; id < outputs->size(); ++id) {
    if ((*outputs)[id]->immutable_name() == "data") {
      Shape data_shape(batch_size, channels, crop_size, crop_size);
      (*outputs)[id]->mutable_shape() = data_shape;
    } else if ((*outputs)[id]->immutable_name() == "label") {
      Shape label_shape(batch_size, 1);
      (*outputs)[id]->mutable_shape() = label_shape;
    }
  }
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);
}  // namespace caffe
