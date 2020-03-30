#include "oneflow/customized/image/random_crop_generator.h"
#include <iostream>

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

CropWindow RandomCropGenerator::GenerateCropWindow(const Shape& shape) {
  CHECK(shape.NumAxes() == 2);
  CropWindow crop;
  int H = shape.At(0);
  int W = shape.At(1);
  if (H <= 0 || W <= 0) { return crop; }

  float min_wh_ratio = aspect_ratio_range_.first;
  float max_wh_ratio = aspect_ratio_range_.second;
  float max_hw_ratio = 1 / aspect_ratio_range_.first;
  float min_area = W * H * area_dis_.a();
  int maxW = std::max<int>(1, static_cast<int>(H * max_wh_ratio));
  int maxH = std::max<int>(1, static_cast<int>(W * max_hw_ratio));

  if (H * maxW < min_area) {
    crop.shape = Shape({H, maxW});
  } else if (W * maxH < min_area) {
    crop.shape = Shape({maxH, W});
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

      crop.shape = Shape({h, w});

      ratio = static_cast<float>(w) / h;

      if (w <= W && h <= H && ratio >= min_wh_ratio && ratio <= max_wh_ratio) { break; }
    }

    if (attempts_left <= 0) {
      float max_area = area_dis_.b() * W * H;
      float ratio = static_cast<float>(W) / H;
      if (ratio > max_wh_ratio) {
        crop.shape = Shape({H, maxW});
      } else if (ratio < min_wh_ratio) {
        crop.shape = Shape({maxH, W});
      } else {
        crop.shape = Shape({H, W});
      }
      float scale = std::min(1.0f, max_area / (crop.shape.At(0) * crop.shape.At(1)));
      crop.shape.Set(0, std::max<int64_t>(1, crop.shape.At(0) * std::sqrt(scale)));
      crop.shape.Set(1, std::max<int64_t>(1, crop.shape.At(1) * std::sqrt(scale)));
    }
  }

  crop.anchor.Set(0, std::uniform_int_distribution<int>(0, H - crop.shape.At(0))(rand_gen_));
  crop.anchor.Set(1, std::uniform_int_distribution<int>(0, W - crop.shape.At(1))(rand_gen_));
  return crop;
}

std::vector<CropWindow> RandomCropGenerator::GenerateCropWindows(const Shape& shape, size_t n) {
  std::seed_seq seq{seed_};
  std::vector<int64_t> seeds(n);
  seq.generate(seeds.begin(), seeds.end());

  std::vector<CropWindow> crop_windows;
  for (std::size_t i = 0; i < n; i++) {
    rand_gen_.seed(seeds.at(i));
    crop_windows.push_back(GenerateCropWindow(shape));
  }
  return crop_windows;
}

}  // namespace oneflow
