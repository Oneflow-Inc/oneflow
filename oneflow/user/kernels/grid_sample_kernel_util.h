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
#ifndef ONEFLOW_USER_KERNELS_GRID_SAMPLE_KERNEL_H_
#define ONEFLOW_USER_KERNELS_GRID_SAMPLE_KERNEL_H_

#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/user/kernels/clip_by_value_kernel.h"
#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#endif  // WITH_CUDA

namespace oneflow {

enum class GridSamplerInterpolation { kBilinear = 0, kNearest, kBicubic };

enum class GridSamplerPadding { kZeros = 0, kBorder, kReflection };

static GridSamplerInterpolation StringToGridSamplerInterpolation(const std::string& mode) {
  if (mode == "bilinear") {
    return GridSamplerInterpolation::kBilinear;
  } else if (mode == "nearest") {
    return GridSamplerInterpolation::kNearest;
  }
  return GridSamplerInterpolation::kBicubic;
}
static GridSamplerPadding StringToGridGridSamplerPadding(const std::string& mode) {
  if (mode == "zeros") {
    return GridSamplerPadding::kZeros;
  } else if (mode == "border") {
    return GridSamplerPadding::kBorder;
  }
  return GridSamplerPadding::kReflection;
}
static bool CanUse32BitIndex(const std::initializer_list<ShapeView>& shapes) {
  for (const auto& shape : shapes) {
    if (shape.elem_cnt() >= std::numeric_limits<int32_t>::max()) { return false; }
  }
  return true;
}

inline int GridSampleGetBlocks(const int64_t number, const int64_t threads_per_block) {
  // Round up division for positive number that cannot cause integer overflow
  auto block_num = (number - 1) / threads_per_block + 1;
  return static_cast<int>(block_num);
}

// This kernel implement is referenced from:
// https://github.com/pytorch/pytorch with git commit id: e7724bb
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/GridSampler.cu
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/GridSampler.cuh

// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t GridSamplerUnnormalize(scalar_t coord, int size,
                                                      bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// GridSamplerUnnormalizeSetGrad works the same as GridSamplerUnnormalize
// except that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t GridSamplerUnnormalizeSetGrad(scalar_t coord, int size,
                                                             bool align_corners,
                                                             scalar_t* grad_in) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t ClipCoordinates(scalar_t in, int clip_limit) {
  return DeviceMin(static_cast<scalar_t>(clip_limit - 1), DeviceMax(in, static_cast<scalar_t>(0)));
}

// ClipCoordinatesSetGrad works similarly to ClipCoordinates except that
// it also returns the `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t ClipCoordinatesSetGrad(scalar_t in, int clip_limit,
                                                      scalar_t* grad_in) {
  // Note that it is important for the gradient calculation that borders
  // are considered out of bounds.
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t ReflectCoordinates(scalar_t in, int twice_low, int twice_high) {
  if (twice_low == twice_high) { return static_cast<scalar_t>(0); }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

// ReflectCoordinatesSetGrad works similarly to ReflectCoordinates except
// that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t ReflectCoordinatesSetGrad(scalar_t in, int twice_low, int twice_high,
                                                         scalar_t* grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  int grad_in_mult_ = 1;
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<scalar_t>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  // `fmod` returns same sign as `in`, which is positive after the `if` above.
  scalar_t extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<scalar_t>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<scalar_t>(-grad_in_mult_);
    return span - extra + min;
  }
}

#if defined(__CUDACC__)
template<typename scalar_t>
static __device__ __forceinline__ scalar_t safe_downgrade_to_int_range(scalar_t x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not WithinBounds2D or WithinBounds3D, and does not cause
  // undefined behavior. See #35506.
  // TODO(pei tingkuan): (explicit or implicit) type conversion from
  // INT_MAX - 1 to float(INT_MAX - 1) indeed changes value from
  // 2147483647 to 2147483648 and losses precision
  // Reference: https://stackoverflow.com/q/526070
  if (x > static_cast<scalar_t>(INT_MAX - 1) || x < INT_MIN || !isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}
#endif

template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t ComputeCoordinates(scalar_t coord, int size,
                                                  GridSamplerPadding padding_mode,
                                                  bool align_corners) {
  if (padding_mode == GridSamplerPadding::kBorder) {
    // clip coordinates to image borders
    coord = ClipCoordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::kReflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = ReflectCoordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = ReflectCoordinates(coord, -1, 2 * size - 1);
    }
    // clip coordinates to image borders
    coord = ClipCoordinates(coord, size);
  }
#if defined(__CUDACC__)
  coord = safe_downgrade_to_int_range(coord);
#endif
  return coord;
}

// Computes the pixel source index value for a grid coordinate
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t GridSamplerComputeSourceIndex(scalar_t coord, int size,
                                                             GridSamplerPadding padding_mode,
                                                             bool align_corners) {
  coord = GridSamplerUnnormalize(coord, size, align_corners);
  coord = ComputeCoordinates(coord, size, padding_mode, align_corners);
  return coord;
}

// GridSamplerComputeSourceIndexSetGrad works similarly to
// GridSamplerComputeSourceIndex except that it also returns the
// `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t GridSamplerComputeSourceIndexSetGrad(scalar_t coord, int size,
                                                                    GridSamplerPadding padding_mode,
                                                                    bool align_corners,
                                                                    scalar_t* grad_in) {
  scalar_t grad_clip, grad_refl;
  coord = GridSamplerUnnormalizeSetGrad(coord, size, align_corners, grad_in);
  if (padding_mode == GridSamplerPadding::kBorder) {
    // clip coordinates to image borders
    coord = ClipCoordinatesSetGrad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == GridSamplerPadding::kReflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = ReflectCoordinatesSetGrad(coord, 0, 2 * (size - 1), &grad_refl);
    } else {
      coord = ReflectCoordinatesSetGrad(coord, -1, 2 * size - 1, &grad_refl);
    }
    // clip coordinates to image borders
    coord = ClipCoordinatesSetGrad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }

#if defined(__CUDACC__)
  coord = safe_downgrade_to_int_range(coord);
#endif
  return coord;
}

static OF_DEVICE_FUNC bool WithinBounds2D(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static OF_DEVICE_FUNC bool WithinBounds3D(int d, int h, int w, int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

template<typename scalar_t>
static OF_DEVICE_FUNC scalar_t GetValueBounded(const scalar_t* data, scalar_t x, scalar_t y, int W,
                                               int H, int sW, int sH,
                                               GridSamplerPadding padding_mode,
                                               bool align_corners) {
  x = ComputeCoordinates(x, W, padding_mode, align_corners);
  y = ComputeCoordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  if (WithinBounds2D(iy, ix, H, W)) { return data[iy * sH + ix * sW]; }
  return static_cast<scalar_t>(0);
}

template<typename scalar_t, typename index_t>
static OF_DEVICE_FUNC void SafeAdd2D(scalar_t* data, int h, int w, int sH, int sW, int H, int W,
                                     scalar_t delta, const index_t NC_offset,
                                     const index_t memory_span) {
  if (WithinBounds2D(h, w, H, W)) {
#if defined(__CUDACC__)
    cuda::atomic::Add(data + NC_offset + h * sH + w * sW, delta);
#else
    data[NC_offset + h * sH + w * sW] += delta;
#endif
  }
}

template<typename scalar_t, typename index_t>
static OF_DEVICE_FUNC void SafeAdd3D(scalar_t* data, int d, int h, int w, int sD, int sH, int sW,
                                     int D, int H, int W, scalar_t delta, const index_t NC_offset,
                                     const index_t memory_span) {
  if (WithinBounds3D(d, h, w, D, H, W)) {
#if defined(__CUDACC__)
    cuda::atomic::Add(data + NC_offset + d * sD + h * sH + w * sW, delta);
#else
    data[NC_offset + d * sD + h * sH + w * sW] += delta;
#endif
  }
}

template<typename scalar_t, typename index_t>
static OF_DEVICE_FUNC void AddValueBounded(scalar_t* data, scalar_t x, scalar_t y, int W, int H,
                                           int sW, int sH, scalar_t delta,
                                           GridSamplerPadding padding_mode, bool align_corners,
                                           const index_t NC_offset, const index_t memory_span) {
  x = ComputeCoordinates(x, W, padding_mode, align_corners);
  y = ComputeCoordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  SafeAdd2D(data, iy, ix, sH, sW, H, W, delta, NC_offset, memory_span);
}

// Calculate the differential of the cubic convolution, i.e. `d coeff / d x`
template<typename scalar_t>
static OF_DEVICE_FUNC void GetCubicCoefficientsGrad(scalar_t coeffs[4], scalar_t t) {
  // Must be the same as forward calculation in
  // aten/src/ATen/native/cuda/UpSample.cuh:get_cubic_upsample_coefficients
  scalar_t A = -0.75;

  scalar_t x;
  x = -1 - t;  // 1 < x = |-1 - tx| < 2
  coeffs[0] = (-3 * A * x - 10 * A) * x - 8 * A;
  x = -t;  // x = |0 - tx| <= 1
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t;  // x = |1 - tx| <= 1
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t;  // 1 < x = |2 - tx| < 2
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template<typename accscalar_t>
OF_DEVICE_FUNC static accscalar_t CubicConvolution1(accscalar_t x, accscalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template<typename accscalar_t>
OF_DEVICE_FUNC static accscalar_t CubicConvolution2(accscalar_t x, accscalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template<typename accscalar_t>
OF_DEVICE_FUNC static void GetCubicUpsamplingCoefficients(accscalar_t coeffs[4], accscalar_t t) {
  accscalar_t A = -0.75;

  accscalar_t x1 = t;
  coeffs[0] = CubicConvolution2<accscalar_t>(x1 + 1.0, A);
  coeffs[1] = CubicConvolution1<accscalar_t>(x1, A);

  // opposite coefficients
  accscalar_t x2 = 1.0 - t;
  coeffs[2] = CubicConvolution1<accscalar_t>(x2, A);
  coeffs[3] = CubicConvolution2<accscalar_t>(x2 + 1.0, A);
}

template<typename scalar_t, typename accscalar_t>
OF_DEVICE_FUNC static accscalar_t cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3,
                                                 accscalar_t t) {
  accscalar_t coeffs[4];
  GetCubicUpsamplingCoefficients<accscalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template<typename data_type, typename index_type>
OF_DEVICE_FUNC void GridSampler4DKernel(const index_type nthreads, const data_type* input_ptr,
                                        const data_type* grid_ptr, data_type* output_ptr,
                                        index_type N, index_type C, index_type inp_H,
                                        index_type inp_W, index_type out_H, index_type out_W,
                                        const GridSamplerInterpolation interpolation_mode,
                                        const GridSamplerPadding padding_mode,
                                        const bool align_corners) {
  index_type inp_sN = C * inp_H * inp_W;
  index_type inp_sC = inp_H * inp_W;
  index_type inp_sH = inp_W;
  index_type inp_sW = 1;
  index_type grid_sN = out_H * out_W * 2;
  index_type grid_sH = out_W * 2;
  index_type grid_sW = 2;
  index_type grid_sCoor = 1;
  index_type out_sN = C * out_H * out_W;
  index_type out_sC = out_H * out_W;
  index_type out_sH = out_W;
  index_type out_sW = 1;

  XPU_1D_KERNEL_LOOP(index, nthreads) {
    const index_type w = index % out_W;
    const index_type h = (index / out_W) % out_H;
    const index_type n = index / (out_H * out_W);
    const index_type grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y co-ordinates from grid
    data_type x = grid_ptr[grid_offset];
    data_type y = grid_ptr[grid_offset + grid_sCoor];

    data_type ix = GridSamplerComputeSourceIndex(x, inp_W, padding_mode, align_corners);
    data_type iy = GridSamplerComputeSourceIndex(y, inp_H, padding_mode, align_corners);

    if (interpolation_mode == GridSamplerInterpolation::kBilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      index_type ix_nw = static_cast<index_type>(::floor(ix));
      index_type iy_nw = static_cast<index_type>(::floor(iy));
      index_type ix_ne = ix_nw + 1;
      index_type iy_ne = iy_nw;
      index_type ix_sw = ix_nw;
      index_type iy_sw = iy_nw + 1;
      index_type ix_se = ix_nw + 1;
      index_type iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      data_type nw = (ix_se - ix) * (iy_se - iy);
      data_type ne = (ix - ix_sw) * (iy_sw - iy);
      data_type sw = (ix_ne - ix) * (iy - iy_ne);
      data_type se = (ix - ix_nw) * (iy - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input_ptr + n * inp_sN;
      auto out_ptr_NCHW = output_ptr + n * out_sN + h * out_sH + w * out_sW;
      for (index_type c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<data_type>(0);
        if (WithinBounds2D(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (WithinBounds2D(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (WithinBounds2D(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (WithinBounds2D(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::kNearest) {
      index_type ix_nearest = static_cast<index_type>(::round(ix));
      index_type iy_nearest = static_cast<index_type>(::round(iy));

      // assign nearest neighor pixel value to output pixel
      auto inp_ptr_NC = input_ptr + n * inp_sN;
      auto out_ptr_NCHW = output_ptr + n * out_sN + h * out_sH + w * out_sW;
      for (index_type c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (WithinBounds2D(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<data_type>(0);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::kBicubic) {
      ix = GridSamplerUnnormalize(x, inp_W, align_corners);
      iy = GridSamplerUnnormalize(y, inp_H, align_corners);

      data_type ix_nw = ::floor(ix);
      data_type iy_nw = ::floor(iy);

      const data_type tx = ix - ix_nw;
      const data_type ty = iy - iy_nw;

      auto inp_ptr_NC = input_ptr + n * inp_sN;
      auto out_ptr_NCHW = output_ptr + n * out_sN + h * out_sH + w * out_sW;
      for (index_type c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        data_type coefficients[4];
#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
        for (index_type i = 0; i < 4; ++i) {
          coefficients[i] = cubic_interp1d(
              GetValueBounded<data_type>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW,
                                         inp_sH, padding_mode, align_corners),
              GetValueBounded<data_type>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW,
                                         inp_sH, padding_mode, align_corners),
              GetValueBounded<data_type>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW,
                                         inp_sH, padding_mode, align_corners),
              GetValueBounded<data_type>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW,
                                         inp_sH, padding_mode, align_corners),
              tx);
        }

        *out_ptr_NCHW =
            cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);
      }
    }
  }
}

template<typename data_type, typename index_type>
OF_DEVICE_FUNC void GridSampler5DKernel(const index_type nthreads, const data_type* input_ptr,
                                        const data_type* grid_ptr, data_type* output_ptr,
                                        index_type N, index_type C, index_type inp_D,
                                        index_type inp_H, index_type inp_W, index_type out_D,
                                        index_type out_H, index_type out_W,
                                        const GridSamplerInterpolation interpolation_mode,
                                        const GridSamplerPadding padding_mode,
                                        const bool align_corners) {
  index_type inp_sN = C * inp_D * inp_H * inp_W;
  index_type inp_sC = inp_D * inp_H * inp_W;
  index_type inp_sD = inp_H * inp_W;
  index_type inp_sH = inp_W;
  index_type inp_sW = 1;
  index_type grid_sN = out_D * out_H * out_W * 3;
  index_type grid_sD = out_H * out_W * 3;
  index_type grid_sH = out_W * 3;
  index_type grid_sW = 3;
  index_type grid_sCoor = 1;
  index_type out_sN = C * out_D * out_H * out_W;
  index_type out_sC = out_D * out_H * out_W;
  index_type out_sD = out_H * out_W;
  index_type out_sH = out_W;
  index_type out_sW = 1;

  XPU_1D_KERNEL_LOOP(index, nthreads) {
    const index_type w = index % out_W;
    const index_type h = (index / out_W) % out_H;
    const index_type d = (index / (out_H * out_W)) % out_D;
    const index_type n = index / (out_D * out_H * out_W);
    const index_type grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    data_type ix = grid_ptr[grid_offset];
    data_type iy = grid_ptr[grid_offset + grid_sCoor];
    data_type iz = grid_ptr[grid_offset + 2 * grid_sCoor];

    ix = GridSamplerComputeSourceIndex(ix, inp_W, padding_mode, align_corners);
    iy = GridSamplerComputeSourceIndex(iy, inp_H, padding_mode, align_corners);
    iz = GridSamplerComputeSourceIndex(iz, inp_D, padding_mode, align_corners);

    if (interpolation_mode == GridSamplerInterpolation::kBilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      index_type ix_tnw = static_cast<index_type>(::floor(ix));
      index_type iy_tnw = static_cast<index_type>(::floor(iy));
      index_type iz_tnw = static_cast<index_type>(::floor(iz));

      index_type ix_tne = ix_tnw + 1;
      index_type iy_tne = iy_tnw;
      index_type iz_tne = iz_tnw;

      index_type ix_tsw = ix_tnw;
      index_type iy_tsw = iy_tnw + 1;
      index_type iz_tsw = iz_tnw;

      index_type ix_tse = ix_tnw + 1;
      index_type iy_tse = iy_tnw + 1;
      index_type iz_tse = iz_tnw;

      index_type ix_bnw = ix_tnw;
      index_type iy_bnw = iy_tnw;
      index_type iz_bnw = iz_tnw + 1;

      index_type ix_bne = ix_tnw + 1;
      index_type iy_bne = iy_tnw;
      index_type iz_bne = iz_tnw + 1;

      index_type ix_bsw = ix_tnw;
      index_type iy_bsw = iy_tnw + 1;
      index_type iz_bsw = iz_tnw + 1;

      index_type ix_bse = ix_tnw + 1;
      index_type iy_bse = iy_tnw + 1;
      index_type iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      data_type tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      data_type tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      data_type tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      data_type tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      data_type bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      data_type bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      data_type bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      data_type bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      auto inp_ptr_NC = input_ptr + n * inp_sN;
      auto out_ptr_NCDHW = output_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (index_type c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
        *out_ptr_NCDHW = static_cast<data_type>(0);
        if (WithinBounds3D(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
        }
        if (WithinBounds3D(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
        }
        if (WithinBounds3D(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
        }
        if (WithinBounds3D(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
        }
        if (WithinBounds3D(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
        }
        if (WithinBounds3D(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
        }
        if (WithinBounds3D(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
        }
        if (WithinBounds3D(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::kNearest) {
      index_type ix_nearest = static_cast<index_type>(::round(ix));
      index_type iy_nearest = static_cast<index_type>(::round(iy));
      index_type iz_nearest = static_cast<index_type>(::round(iz));

      // assign nearest neighor pixel value to output pixel
      auto inp_ptr_NC = input_ptr + n * inp_sN;
      auto out_ptr_NCDHW = output_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (index_type c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        if (WithinBounds3D(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW =
              inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCDHW = static_cast<data_type>(0);
        }
      }
    }
  }
}

// Note [Passing pointer and offset to fastAtomicAdd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For its internal bounds checking, fastAtomicAdd needs to know where the destination address
// lies relative to the entire tensor, so we pass the base grad_input_ptr and full offset
// information, including batch * channel offset (NC_offset).

template<typename data_type, typename index_type>
OF_DEVICE_FUNC void GridSampler4DBackwardKernel(
    const index_type nthreads, const data_type* grad_output_ptr, const data_type* input_ptr,
    const data_type* grid_ptr, data_type* grad_input_ptr, data_type* grad_grid_ptr, index_type N,
    index_type C, index_type inp_H, index_type inp_W, index_type out_H, index_type out_W,
    const GridSamplerInterpolation interpolation_mode, const GridSamplerPadding padding_mode,
    const bool align_corners, const index_type grad_input_memory_span) {
  index_type inp_sN = C * inp_H * inp_W;
  index_type inp_sC = inp_H * inp_W;
  index_type inp_sH = inp_W;
  index_type inp_sW = 1;
  index_type grid_sN = out_H * out_W * 2;
  index_type grid_sH = out_W * 2;
  index_type grid_sW = 2;
  index_type grid_sCoor = 1;
  index_type gOut_sN = C * out_H * out_W;
  index_type gOut_sC = out_H * out_W;
  index_type gOut_sH = out_W;
  index_type gOut_sW = 1;
  index_type gInp_sN = inp_sN;
  index_type gInp_sC = inp_sC;
  index_type gInp_sH = inp_sH;
  index_type gInp_sW = inp_sW;
  index_type gGrid_sW = grid_sW;

  XPU_1D_KERNEL_LOOP(index, nthreads) {
    const index_type w = index % out_W;
    const index_type h = (index / out_W) % out_H;
    const index_type n = index / (out_H * out_W);
    const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y co-ordinates from grid
    data_type x = grid_ptr[grid_offset];
    data_type y = grid_ptr[grid_offset + grid_sCoor];

    // multipliers for gradients on ix and iy
    data_type gix_mult, giy_mult;
    data_type ix =
        GridSamplerComputeSourceIndexSetGrad(x, inp_W, padding_mode, align_corners, &gix_mult);
    data_type iy =
        GridSamplerComputeSourceIndexSetGrad(y, inp_H, padding_mode, align_corners, &giy_mult);

    if (interpolation_mode == GridSamplerInterpolation::kBilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      index_type ix_nw = static_cast<index_type>(::floor(ix));
      index_type iy_nw = static_cast<index_type>(::floor(iy));
      index_type ix_ne = ix_nw + 1;
      index_type iy_ne = iy_nw;
      index_type ix_sw = ix_nw;
      index_type iy_sw = iy_nw + 1;
      index_type ix_se = ix_nw + 1;
      index_type iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      data_type nw = (ix_se - ix) * (iy_se - iy);
      data_type ne = (ix - ix_sw) * (iy_sw - iy);
      data_type sw = (ix_ne - ix) * (iy - iy_ne);
      data_type se = (ix - ix_nw) * (iy - iy_nw);

      data_type gix = static_cast<data_type>(0), giy = static_cast<data_type>(0);
      const data_type* gOut_ptr_NCHW = grad_output_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      index_type NC_offset = n * gInp_sN;
      const data_type* inp_ptr_NC = input_ptr + n * inp_sN;
      for (index_type c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
        data_type gOut = *gOut_ptr_NCHW;

        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        SafeAdd2D(grad_input_ptr, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut,
                  NC_offset, grad_input_memory_span);
        SafeAdd2D(grad_input_ptr, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut,
                  NC_offset, grad_input_memory_span);
        SafeAdd2D(grad_input_ptr, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut,
                  NC_offset, grad_input_memory_span);
        SafeAdd2D(grad_input_ptr, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut,
                  NC_offset, grad_input_memory_span);

        // calculate grad_grid
        if (WithinBounds2D(iy_nw, ix_nw, inp_H, inp_W)) {
          data_type nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
          gix -= nw_val * (iy_se - iy) * gOut;
          giy -= nw_val * (ix_se - ix) * gOut;
        }
        if (WithinBounds2D(iy_ne, ix_ne, inp_H, inp_W)) {
          data_type ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
          gix += ne_val * (iy_sw - iy) * gOut;
          giy -= ne_val * (ix - ix_sw) * gOut;
        }
        if (WithinBounds2D(iy_sw, ix_sw, inp_H, inp_W)) {
          data_type sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
          gix -= sw_val * (iy - iy_ne) * gOut;
          giy += sw_val * (ix_ne - ix) * gOut;
        }
        if (WithinBounds2D(iy_se, ix_se, inp_H, inp_W)) {
          data_type se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
          gix += se_val * (iy - iy_nw) * gOut;
          giy += se_val * (ix - ix_nw) * gOut;
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
      //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      data_type* gGrid_ptr_NHW = grad_grid_ptr + index * gGrid_sW;
      gGrid_ptr_NHW[0] = gix_mult * gix;
      gGrid_ptr_NHW[1] = giy_mult * giy;
    } else if (interpolation_mode == GridSamplerInterpolation::kNearest) {
      index_type ix_nearest = static_cast<index_type>(::round(ix));
      index_type iy_nearest = static_cast<index_type>(::round(iy));

      // assign nearest neighor pixel value to output pixel
      const data_type* gOut_ptr_NCHW = grad_output_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      index_type NC_offset = n * gInp_sN;
      for (index_type c = 0; c < C; ++c, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        SafeAdd2D(grad_input_ptr, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W,
                  *gOut_ptr_NCHW, NC_offset, grad_input_memory_span);
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
      //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      data_type* gGrid_ptr_NHW = grad_grid_ptr + index * gGrid_sW;
      gGrid_ptr_NHW[0] = static_cast<data_type>(0);
      gGrid_ptr_NHW[1] = static_cast<data_type>(0);
    } else if (interpolation_mode == GridSamplerInterpolation::kBicubic) {
      ix = GridSamplerUnnormalizeSetGrad(x, inp_W, align_corners, &gix_mult);
      iy = GridSamplerUnnormalizeSetGrad(y, inp_H, align_corners, &giy_mult);

      data_type ix_nw = ::floor(ix);
      data_type iy_nw = ::floor(iy);

      const data_type tx = ix - ix_nw;
      const data_type ty = iy - iy_nw;

      data_type x_coeffs[4];
      data_type y_coeffs[4];
      data_type x_coeffs_grad[4];
      data_type y_coeffs_grad[4];

      GetCubicUpsamplingCoefficients<data_type>(x_coeffs, tx);
      GetCubicUpsamplingCoefficients<data_type>(y_coeffs, ty);
      GetCubicCoefficientsGrad<data_type>(x_coeffs_grad, tx);
      GetCubicCoefficientsGrad<data_type>(y_coeffs_grad, ty);

      data_type gix = static_cast<data_type>(0);
      data_type giy = static_cast<data_type>(0);

      const data_type* gOut_ptr_NCHW = grad_output_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      index_type NC_offset = n * gInp_sN;
      const data_type* inp_ptr_NC = input_ptr + n * inp_sN;

      for (index_type c = 0; c < C;
           ++c, gOut_ptr_NCHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC) {
        data_type gOut = *gOut_ptr_NCHW;

#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
        for (index_type i = 0; i < 4; ++i) {
#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
          for (index_type j = 0; j < 4; ++j) {
            // set input gradient. See Note [Passing pointer and offset to fastAtomicAdd].
            AddValueBounded<data_type>(grad_input_ptr, ix_nw - 1 + i, iy_nw - 1 + j, inp_W, inp_H,
                                       gInp_sW, gInp_sH, gOut * x_coeffs[i] * y_coeffs[j],
                                       padding_mode, align_corners, NC_offset,
                                       grad_input_memory_span);

            // set grid gradient
            data_type val =
                GetValueBounded<data_type>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j, inp_W, inp_H,
                                           inp_sW, inp_sH, padding_mode, align_corners);

            gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
            giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
          }
        }
      }

      data_type* gGrid_ptr_NHW = grad_grid_ptr + index * gGrid_sW;
      gGrid_ptr_NHW[0] = gix_mult * gix;
      gGrid_ptr_NHW[1] = giy_mult * giy;
    }
  }
}

template<typename data_type, typename index_type>
OF_DEVICE_FUNC void GridSampler5DBackwardKernel(
    const index_type nthreads, const data_type* grad_output_ptr, const data_type* input_ptr,
    const data_type* grid_ptr, data_type* grad_input_ptr, data_type* grad_grid_ptr, index_type N,
    index_type C, index_type inp_D, index_type inp_H, index_type inp_W, index_type out_D,
    index_type out_H, index_type out_W, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners,
    const index_type grad_input_memory_span) {
  index_type inp_sN = C * inp_D * inp_H * inp_W;
  index_type inp_sC = inp_D * inp_H * inp_W;
  index_type inp_sD = inp_H * inp_W;
  index_type inp_sH = inp_W;
  index_type inp_sW = 1;
  index_type grid_sN = out_D * out_H * out_W * 3;
  index_type grid_sD = out_H * out_W * 3;
  index_type grid_sH = out_W * 3;
  index_type grid_sW = 3;
  index_type grid_sCoor = 1;
  index_type gOut_sN = C * out_D * out_H * out_W;
  index_type gOut_sC = out_D * out_H * out_W;
  index_type gOut_sD = out_H * out_W;
  index_type gOut_sH = out_W;
  index_type gOut_sW = 1;
  index_type gInp_sN = inp_sN;
  index_type gInp_sC = inp_sC;
  index_type gInp_sD = inp_sD;
  index_type gInp_sH = inp_sH;
  index_type gInp_sW = inp_sW;
  index_type gGrid_sW = grid_sW;

  XPU_1D_KERNEL_LOOP(index, nthreads) {
    const index_type w = index % out_W;
    const index_type h = (index / out_W) % out_H;
    const index_type d = (index / (out_H * out_W)) % out_D;
    const index_type n = index / (out_D * out_H * out_W);
    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    data_type ix = grid_ptr[grid_offset];
    data_type iy = grid_ptr[grid_offset + grid_sCoor];
    data_type iz = grid_ptr[grid_offset + 2 * grid_sCoor];

    // multipliers for gradients on ix, iy, and iz
    data_type gix_mult, giy_mult, giz_mult;
    ix = GridSamplerComputeSourceIndexSetGrad(ix, inp_W, padding_mode, align_corners, &gix_mult);
    iy = GridSamplerComputeSourceIndexSetGrad(iy, inp_H, padding_mode, align_corners, &giy_mult);
    iz = GridSamplerComputeSourceIndexSetGrad(iz, inp_D, padding_mode, align_corners, &giz_mult);

    if (interpolation_mode == GridSamplerInterpolation::kBilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      index_type ix_tnw = static_cast<index_type>(::floor(ix));
      index_type iy_tnw = static_cast<index_type>(::floor(iy));
      index_type iz_tnw = static_cast<index_type>(::floor(iz));

      index_type ix_tne = ix_tnw + 1;
      index_type iy_tne = iy_tnw;
      index_type iz_tne = iz_tnw;

      index_type ix_tsw = ix_tnw;
      index_type iy_tsw = iy_tnw + 1;
      index_type iz_tsw = iz_tnw;

      index_type ix_tse = ix_tnw + 1;
      index_type iy_tse = iy_tnw + 1;
      index_type iz_tse = iz_tnw;

      index_type ix_bnw = ix_tnw;
      index_type iy_bnw = iy_tnw;
      index_type iz_bnw = iz_tnw + 1;

      index_type ix_bne = ix_tnw + 1;
      index_type iy_bne = iy_tnw;
      index_type iz_bne = iz_tnw + 1;

      index_type ix_bsw = ix_tnw;
      index_type iy_bsw = iy_tnw + 1;
      index_type iz_bsw = iz_tnw + 1;

      index_type ix_bse = ix_tnw + 1;
      index_type iy_bse = iy_tnw + 1;
      index_type iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      data_type tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      data_type tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      data_type tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      data_type tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      data_type bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      data_type bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      data_type bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      data_type bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      data_type gix = static_cast<data_type>(0), giy = static_cast<data_type>(0),
                giz = static_cast<data_type>(0);
      const data_type* gOut_ptr_NCDHW =
          grad_output_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
      index_type NC_offset = n * gInp_sN;
      const data_type* inp_ptr_NC = input_ptr + n * inp_sN;
      // calculate bilinear weighted pixel value and set output pixel
      for (index_type c = 0; c < C;
           ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC) {
        data_type gOut = *gOut_ptr_NCDHW;

        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        SafeAdd3D(grad_input_ptr, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, tnw * gOut, NC_offset, grad_input_memory_span);
        SafeAdd3D(grad_input_ptr, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, tne * gOut, NC_offset, grad_input_memory_span);
        SafeAdd3D(grad_input_ptr, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, tsw * gOut, NC_offset, grad_input_memory_span);
        SafeAdd3D(grad_input_ptr, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, tse * gOut, NC_offset, grad_input_memory_span);
        SafeAdd3D(grad_input_ptr, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, bnw * gOut, NC_offset, grad_input_memory_span);
        SafeAdd3D(grad_input_ptr, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, bne * gOut, NC_offset, grad_input_memory_span);
        SafeAdd3D(grad_input_ptr, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, bsw * gOut, NC_offset, grad_input_memory_span);
        SafeAdd3D(grad_input_ptr, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H,
                  inp_W, bse * gOut, NC_offset, grad_input_memory_span);

        // calculate grad_grid
        if (WithinBounds3D(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          data_type tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
          gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
          giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
          giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
        }
        if (WithinBounds3D(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          data_type tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
          gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
          giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
          giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
        }
        if (WithinBounds3D(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          data_type tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
          gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
          giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
          giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
        }
        if (WithinBounds3D(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          data_type tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
          gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
          giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
          giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
        }
        if (WithinBounds3D(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          data_type bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
          gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
          giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
          giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
        }
        if (WithinBounds3D(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          data_type bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
          gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
          giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
          giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
        }
        if (WithinBounds3D(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          data_type bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
          gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
          giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
          giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
        }
        if (WithinBounds3D(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          data_type bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
          gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
          giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
          giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
      //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
      data_type* gGrid_ptr_NDHW = grad_grid_ptr + index * gGrid_sW;
      gGrid_ptr_NDHW[0] = gix_mult * gix;
      gGrid_ptr_NDHW[1] = giy_mult * giy;
      gGrid_ptr_NDHW[2] = giz_mult * giz;
    } else if (interpolation_mode == GridSamplerInterpolation::kNearest) {
      auto ix_nearest = static_cast<index_type>(::round(ix));
      auto iy_nearest = static_cast<index_type>(::round(iy));
      auto iz_nearest = static_cast<index_type>(::round(iz));

      // assign nearest neighor pixel value to output pixel
      const data_type* gOut_ptr_NCDHW =
          grad_output_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
      index_type NC_offset = n * gInp_sN;
      for (index_type c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC) {
        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        SafeAdd3D(grad_input_ptr, iz_nearest, iy_nearest, ix_nearest, gInp_sD, gInp_sH, gInp_sW,
                  inp_D, inp_H, inp_W, *gOut_ptr_NCDHW, NC_offset, grad_input_memory_span);
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
      //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
      data_type* gGrid_ptr_NDHW = grad_grid_ptr + index * gGrid_sW;
      gGrid_ptr_NDHW[0] = static_cast<data_type>(0);
      gGrid_ptr_NDHW[1] = static_cast<data_type>(0);
      gGrid_ptr_NDHW[2] = static_cast<data_type>(0);
    }
  }
}

template<DeviceType device_type, typename data_type, typename index_type>
struct GridSampleKernelUtil final {
  static void Forward4D(user_op::KernelComputeContext* ctx, const user_op::Tensor* input,
                        const user_op::Tensor* grid, user_op::Tensor* output,
                        GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                        const bool align_corners, const ShapeView& input_shape,
                        const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count);
  static void Forward5D(user_op::KernelComputeContext* ctx, const user_op::Tensor* input,
                        const user_op::Tensor* grid, user_op::Tensor* output,
                        GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                        const bool align_corners, const ShapeView& input_shape,
                        const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count);

  static void Backward4D(user_op::KernelComputeContext* ctx, const user_op::Tensor* doutput,
                         const user_op::Tensor* input, const user_op::Tensor* grid,
                         user_op::Tensor* dinput, user_op::Tensor* dgrid,
                         GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                         const bool align_corners, const ShapeView& input_shape,
                         const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count);
  static void Backward5D(user_op::KernelComputeContext* ctx, const user_op::Tensor* doutput,
                         const user_op::Tensor* input, const user_op::Tensor* grid,
                         user_op::Tensor* dinput, user_op::Tensor* dgrid,
                         GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                         const bool align_corners, const ShapeView& input_shape,
                         const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count);
};

// macros for functors instantiate(used by grid_sample_kernel_util.cu, grid_sample_kernel_util.cpp)
#define INSTANTIATE_GRID_SAMPLE_KERNEL_UTIL(device_type, dtype_pair, itype_pair)  \
  template struct GridSampleKernelUtil<device_type, OF_PP_PAIR_FIRST(dtype_pair), \
                                       OF_PP_PAIR_FIRST(itype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_GRID_SAMPLE_KERNEL_H_
