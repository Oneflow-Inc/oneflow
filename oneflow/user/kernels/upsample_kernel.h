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
#include "oneflow/core/common/nd_index_offset_helper.h"

OF_DEVICE_FUNC static int64_t GetNearestInputIndexFunc(const int64_t out_dim_idx, const float scale,
                                                       const int64_t in_dim_size) {
  int64_t index = static_cast<int64_t>(std::floor((static_cast<float>(out_dim_idx) * scale)));
  index = index > in_dim_size - 1 ? in_dim_size - 1 : index;
  index = index < static_cast<int64_t>(0) ? static_cast<int64_t>(0) : index;
  return index;
}

template<typename T>
OF_DEVICE_FUNC T GetAreaPixelScaleFunc(const int64_t input_size, const int64_t output_size,
                                       bool align_corners, const T scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    } else {
      return 0;
    }
  } else {
    return (scale > 0. ? 1.0 / scale : static_cast<T>(input_size) / output_size);
  }
}

template<typename T>
struct BilinearParam {
  int64_t top_h_index;
  int64_t bottom_h_index;
  int64_t left_w_index;
  int64_t right_w_index;
  T w_lerp;
  T h_lerp;
};

template<typename T>
OF_DEVICE_FUNC void GetBilinearParamFunc(const bool align_corners, const int64_t h, const int64_t w,
                                         const int64_t in_height, const int64_t in_width,
                                         const T scale_h, const T scale_w,
                                         BilinearParam<T>* params) {
  T h1r;
  if (align_corners) {
    h1r = scale_h * static_cast<T>(h);
  } else {
    h1r = (static_cast<T>(h) + 0.5f) * scale_h - 0.5f;
    h1r = h1r < 0 ? 0 : h1r;
  }
  const int64_t h1 = h1r;
  const int64_t h1p = (h1 < in_height - 1) ? 1 : 0;

  T w1r;
  if (align_corners) {
    w1r = scale_w * static_cast<T>(w);
  } else {
    w1r = (static_cast<T>(w) + 0.5f) * scale_w - 0.5f;
    w1r = w1r < 0 ? 0 : w1r;
  }
  const int64_t w1 = w1r;
  const int64_t w1p = (w1 < in_width - 1) ? 1 : 0;

  params->top_h_index = h1;
  params->bottom_h_index = h1 + h1p;
  params->h_lerp = h1r - h1;
  params->left_w_index = w1;
  params->right_w_index = w1 + w1p;
  params->w_lerp = w1r - w1;
}
