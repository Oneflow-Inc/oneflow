#ifndef ONEFLOW_CUSTOMIZED_IMAGE_RANDOM_CROP_GENERATOR_H_
#define ONEFLOW_CUSTOMIZED_IMAGE_RANDOM_CROP_GENERATOR_H_

#include "oneflow/customized/image/crop_window.h"

namespace oneflow {

using AspectRatioRange = std::pair<float, float>;
using AreaRange = std::pair<float, float>;

class RandomCropGenerator {
 public:
  RandomCropGenerator(AspectRatioRange aspect_ratio_range, AreaRange area_range, int64_t seed,
                      int32_t num_attempts);

  void GenerateCropWindow(const Shape& shape, CropWindow* crop_window);
  void GenerateCropWindows(const Shape& shape, size_t n, std::vector<CropWindow>* crop_windows);

 private:
  AspectRatioRange aspect_ratio_range_;
  std::uniform_real_distribution<float> aspect_ratio_log_dis_;
  std::uniform_real_distribution<float> area_dis_;
  std::mt19937 rand_gen_;
  int64_t seed_;
  int32_t num_attempts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_IMAGE_RANDOM_CROP_GENERATOR_H_
