#include <cstdint>
#include <vector>
#include "layers/data_layers.h"
#include "common/common.h"
#include "layers/layer_factory.h"

namespace caffe {
  template <typename Dtype>
  void DataPreloadLayer<Dtype>::LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs,
    uint32_t preload_size) {
    CHECK_GT(preload_size, 0) << "Preload size should be greater than zero.";
    preload_size_ = preload_size;
    DLOG(INFO) << "Settingup DataPreloadLayer...";
  }
  template <typename Dtype>
  void DataPreloadLayer<Dtype>::Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs) {
    CHECK_EQ(inputs.size(), 0);   // no other denpendencies
    CHECK_EQ(outputs->size(), 2);  // data and label
    int64_t batch_size = data_param_.batch_size();
    CHECK(data_param_.has_transform_param());
    int64_t channels = data_param_.transform_param().channel();
    int64_t crop_size = data_param_.transform_param().crop_size();

    // NOTE(Jiahui Yu): Output is rawdata, the size must be one.
    CHECK_EQ(outputs->size(), 1)
      << "The output size of data_preload_layer must be 1";
    if ((*outputs)[0]->immutable_name() == "rawdata") {
      // TODO(Jiahui Yu): The size need to be changed.
      Shape data_shape(batch_size * preload_size_,
        channels, crop_size, crop_size);
      (*outputs)[0]->mutable_shape() = data_shape;
    } else {
      CHECK(false) << "the output name must be rawdata";
    }
  }

  INSTANTIATE_CLASS(DataPreloadLayer);
  REGISTER_LAYER_CLASS(DataPreload);
}  // namespace caffe
