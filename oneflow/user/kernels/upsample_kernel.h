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

OF_DEVICE_FUNC static int64_t GetLinearInputIndex(const int64_t out_dim_idx, const float scale,
                                                  bool align_corners) {
  if (align_corners) {
    return scale * out_dim_idx;
  } else {
    int64_t src_idx = static_cast<int64_t>(scale * (out_dim_idx + 0.5) - 0.5);
    return src_idx < 0 ? 0 : src_idx;
  }
}

OF_DEVICE_FUNC static int64_t GetNearestInputIndex(const int64_t out_dim_idx, const float scale,
                                                   const int64_t in_dim_size) {
  int64_t index = static_cast<int64_t>(std::floor((static_cast<float>(out_dim_idx) * scale)));
  index = index > in_dim_size - 1 ? in_dim_size - 1 : index;
  index = index < static_cast<int64_t>(0) ? static_cast<int64_t>(0) : index;
  return index;
}

template<typename T>
OF_DEVICE_FUNC T GetAreaPixelScale(const int64_t input_size, const int64_t output_size,
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
OF_DEVICE_FUNC void GetBilinearParam(const bool align_corners, const int64_t h, const int64_t w,
                                     const int64_t in_height, const int64_t in_width,
                                     const T scale_h, const T scale_w, BilinearParam<T>* params) {
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

template <typename T>
OF_DEVICE_FUNC T upsample_get_value_bounded(
    T* data,
    int64_t width,
    int64_t height,
    int64_t x,
    int64_t y) {
  int64_t access_x = std::max(std::min(x, width - 1), static_cast<int64_t>(0));
  int64_t access_y = std::max(std::min(y, height - 1), static_cast<int64_t>(0));
  return data[access_y * width + access_x];
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename T>
OF_DEVICE_FUNC T cubic_convolution1(T x, T A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename T>
OF_DEVICE_FUNC T cubic_convolution2(T x, T A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename T>
OF_DEVICE_FUNC void get_cubic_upsample_coefficients(
    T coeffs[4],
    T t) {
  T A = -0.75;

  T x1 = t;
  coeffs[0] = cubic_convolution2<T>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<T>(x1, A);

  // opposite coefficients
  T x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<T>(x2, A);
  coeffs[3] = cubic_convolution2<T>(x2 + 1.0, A);
}

template <typename T>
OF_DEVICE_FUNC T cubic_interp1d(
    T x0,
    T x1,
    T x2,
    T x3,
    T t) {
  T coeffs[4];
  get_cubic_upsample_coefficients<T>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}
