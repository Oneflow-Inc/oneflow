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
#include <math.h>

OF_DEVICE_FUNC double GetLinearInputIndex(const int64_t out_dim_idx, const double scale,
                                          bool align_corners) {
  if (align_corners) {
    return static_cast<double>(scale * out_dim_idx);
  } else {
    double src_idx = scale * (out_dim_idx + 0.5) - 0.5;
    return static_cast<double>(src_idx < 0 ? 0 : src_idx);
  }
}

OF_DEVICE_FUNC static int64_t GetNearestInputIndex(const int64_t out_dim_idx, const double scale,
                                                   const int64_t in_dim_size) {
  int64_t index = static_cast<int64_t>(floorf(out_dim_idx * scale));
  index = index > in_dim_size - 1 ? in_dim_size - 1 : index;
  return index;
}

OF_DEVICE_FUNC double GetAreaPixelScale(const int64_t input_size, const int64_t output_size,
                                        bool align_corners, const double scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<double>(input_size - 1) / (output_size - 1);
    } else {
      return 0;
    }
  } else {
    return (scale > 0. ? 1.0 / scale : static_cast<double>(input_size) / output_size);
  }
}

OF_DEVICE_FUNC double GetAreaPixel(const double scale, const int64_t dst_index, bool align_corners,
                                   bool cubic = false) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    double src_idx = scale * (dst_index + 0.5) - 0.5;
    return (!cubic && src_idx < 0) ? static_cast<double>(0) : src_idx;
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
                                     const double scale_h, const double scale_w,
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

template<typename T>
OF_DEVICE_FUNC void upsample_increment_value_bounded(T* data, int64_t width, int64_t height,
                                                     int64_t x, int64_t y, T value) {
  int64_t access_x = std::max(std::min(x, width - 1), static_cast<int64_t>(0));
  int64_t access_y = std::max(std::min(y, height - 1), static_cast<int64_t>(0));
  data[access_y * width + access_x] += value;
}

template<typename T>
OF_DEVICE_FUNC T upsample_get_value_bounded(const T* data, const int64_t width,
                                            const int64_t height, const int64_t x,
                                            const int64_t y) {
  int64_t access_x = x;
  access_x = access_x > width - 1 ? width - 1 : access_x;
  access_x = access_x < 0 ? 0 : access_x;

  int64_t access_y = y;
  access_y = access_y > height - 1 ? height - 1 : access_y;
  access_y = access_y < 0 ? 0 : access_y;

  return data[access_y * width + access_x];
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm

template<typename T>
OF_DEVICE_FUNC T cubic_convolution1(const T x, const T A) {
  return ((A + static_cast<T>(2.0)) * x - (A + static_cast<T>(3.0))) * x * x + static_cast<T>(1.0);
}

template<typename T>
OF_DEVICE_FUNC T cubic_convolution2(const T x, const T A) {
  return ((A * x - static_cast<T>(5.0) * A) * x + static_cast<T>(8.0) * A) * x
         - static_cast<T>(4.0) * A;
}

template<typename T>
OF_DEVICE_FUNC void get_cubic_upsample_coefficients(T coeffs[4], const T t) {
  T A = -0.75;

  T x1 = t;
  coeffs[0] = cubic_convolution2<T>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<T>(x1, A);

  // opposite coefficients
  T x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<T>(x2, A);
  coeffs[3] = cubic_convolution2<T>(x2 + 1.0, A);
}

template<typename T>
OF_DEVICE_FUNC T cubic_interp1d(const T x0, const T x1, const T x2, const T x3, const T t) {
  T coeffs[4];
  get_cubic_upsample_coefficients<T>(coeffs, t);
  return x0 * coeffs[0] * 1.0 + x1 * coeffs[1] * 1.0 + x2 * coeffs[2] * 1.0 + x3 * coeffs[3] * 1.0;
}
