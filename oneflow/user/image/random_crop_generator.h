/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_USER_IMAGE_RANDOM_CROP_GENERATOR_H_
#define ONEFLOW_USER_IMAGE_RANDOM_CROP_GENERATOR_H_

#include "oneflow/user/image/crop_window.h"

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

#endif  // ONEFLOW_USER_IMAGE_RANDOM_CROP_GENERATOR_H_
