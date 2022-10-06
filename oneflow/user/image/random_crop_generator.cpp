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
#include "oneflow/user/image/random_crop_generator.h"

namespace oneflow {

RandomCropGenerator::RandomCropGenerator(AspectRatioRange aspect_ratio_range, AreaRange area_range,
                                         int64_t seed, int32_t num_attempts)
    : aspect_ratio_range_(aspect_ratio_range),
      aspect_ratio_log_dis_(std::log(aspect_ratio_range.first),
                            std::log(aspect_ratio_range.second)),
      area_dis_(area_range.first, area_range.second),
      rand_gen_(seed),
      seed_(seed),
      num_attempts_(num_attempts) {}

void RandomCropGenerator::GenerateCropWindow(const Shape& shape, CropWindow* crop_window) {
  CHECK_EQ(shape.NumAxes(), 2);
  CHECK(crop_window != nullptr);

  int H = shape.At(0);
  int W = shape.At(1);
  if (H <= 0 || W <= 0) { return; }

  float min_wh_ratio = aspect_ratio_range_.first;
  float max_wh_ratio = aspect_ratio_range_.second;
  float max_hw_ratio = 1 / aspect_ratio_range_.first;
  float min_area = W * H * area_dis_.a();
  int maxW = std::max<int>(1, static_cast<int>(H * max_wh_ratio));
  int maxH = std::max<int>(1, static_cast<int>(W * max_hw_ratio));

  if (H * maxW < min_area) {
    crop_window->shape = Shape({H, maxW});
  } else if (W * maxH < min_area) {
    crop_window->shape = Shape({maxH, W});
  } else {
    int attempts_left = num_attempts_;
    for (; attempts_left > 0; attempts_left--) {
      float scale = area_dis_(rand_gen_);

      size_t original_area = H * W;
      float target_area = scale * original_area;

      float ratio = std::exp(aspect_ratio_log_dis_(rand_gen_));
      int w = static_cast<int>(std::roundf(sqrtf(target_area * ratio)));
      int h = static_cast<int>(std::roundf(sqrtf(target_area / ratio)));

      w = std::max(w, 1);
      h = std::max(h, 1);

      crop_window->shape = Shape({h, w});

      ratio = static_cast<float>(w) / h;

      if (w <= W && h <= H && ratio >= min_wh_ratio && ratio <= max_wh_ratio) { break; }
    }

    if (attempts_left <= 0) {
      float max_area = area_dis_.b() * W * H;
      float ratio = static_cast<float>(W) / H;
      if (ratio > max_wh_ratio) {
        crop_window->shape = Shape({H, maxW});
      } else if (ratio < min_wh_ratio) {
        crop_window->shape = Shape({maxH, W});
      } else {
        crop_window->shape = Shape({H, W});
      }
      float scale =
          std::min(1.0f, max_area / (crop_window->shape.At(0) * crop_window->shape.At(1)));
      crop_window->shape.Set(0, std::max<int64_t>(1, crop_window->shape.At(0) * std::sqrt(scale)));
      crop_window->shape.Set(1, std::max<int64_t>(1, crop_window->shape.At(1) * std::sqrt(scale)));
    }
  }

  crop_window->anchor.Set(
      0, std::uniform_int_distribution<int>(0, H - crop_window->shape.At(0))(rand_gen_));
  crop_window->anchor.Set(
      1, std::uniform_int_distribution<int>(0, W - crop_window->shape.At(1))(rand_gen_));
}

void RandomCropGenerator::GenerateCropWindows(const Shape& shape, size_t n,
                                              std::vector<CropWindow>* crop_windows) {
  std::seed_seq seq{seed_};
  std::vector<int64_t> seeds(n);
  seq.generate(seeds.begin(), seeds.end());
  crop_windows->resize(n);

  for (std::size_t i = 0; i < n; i++) {
    rand_gen_.seed(seeds.at(i));
    GenerateCropWindow(shape, &(crop_windows->at(i)));
  }
}

}  // namespace oneflow
