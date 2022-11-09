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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__device__ T BilinearInterpolate(const T* channel_dptr, const int32_t height, const int32_t width,
                                 T y, T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) { return 0; }

  if (y <= 0) { y = 0; }
  if (x <= 0) { x = 0; }
  int32_t y_low = static_cast<int32_t>(y);
  int32_t x_low = static_cast<int32_t>(x);
  int32_t y_high = 0;
  int32_t x_high = 0;

  if (y_low >= height - 1) {
    y_low = height - 1;
    y_high = y_low;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_low = width - 1;
    x_high = x_low;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }

  const T ly = y - y_low;
  const T lx = x - x_low;
  const T hy = 1.f - ly;
  const T hx = 1.f - lx;

  // https://en.wikipedia.org/wiki/Bilinear_interpolation
  const int64_t q11 = y_low * width + x_low;
  const int64_t q21 = y_low * width + x_high;
  const int64_t q12 = y_high * width + x_low;
  const int64_t q22 = y_high * width + x_high;
  //  no 1 / (x_high - x_low) * (y_high - y_low) because it will always be 1 in RoI Align
  return (hy * hx) * channel_dptr[q11] + (hy * lx) * channel_dptr[q21]
         + (ly * hx) * channel_dptr[q12] + (ly * lx) * channel_dptr[q22];
}

template<typename T>
__device__ bool BilinearInterpolateDiff(const T bin_diff_avg, const int64_t height,
                                        const int64_t width, T y, T x, T& diff11, T& diff21,
                                        T& diff12, T& diff22, int32_t& x_low, int32_t& x_high,
                                        int32_t& y_low, int32_t& y_high) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) { return false; }

  if (y <= 0) { y = 0; }
  if (x <= 0) { x = 0; }

  y_low = static_cast<int32_t>(y);
  x_low = static_cast<int32_t>(x);

  if (y_low >= height - 1) {
    y_low = height - 1;
    y_high = y_low;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_low = width - 1;
    x_high = x_low;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }

  const T ly = y - y_low;
  const T lx = x - x_low;
  const T hy = 1.f - ly;
  const T hx = 1.f - lx;

  diff11 = bin_diff_avg * hy * hx;
  diff21 = bin_diff_avg * hy * lx;
  diff12 = bin_diff_avg * ly * hx;
  diff22 = bin_diff_avg * ly * lx;
  return true;
}

template<typename T>
__global__ void RoiAlignForward(const int64_t nthreads, const T* in_dptr, const T* rois_dptr,
                                const T spatial_scale, const int32_t sampling_ratio,
                                const int64_t channel_num, const int64_t height,
                                const int64_t width, const int64_t pooled_height,
                                const int64_t pooled_width, const bool aligned, T* out_dptr) {
  const int64_t pooled_area = pooled_height * pooled_width;
  const int64_t channel_pooled_area = channel_num * pooled_height * pooled_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = index / channel_pooled_area;
    const T* offset_rois_dptr = rois_dptr + r * 5;
    const int64_t n = static_cast<int64_t>(offset_rois_dptr[0]);
    const T align_offset = aligned ? static_cast<T>(0.5) : static_cast<T>(0.f);
    const T roi_start_w = offset_rois_dptr[1] * spatial_scale - align_offset;
    const T roi_start_h = offset_rois_dptr[2] * spatial_scale - align_offset;
    const T roi_end_w = offset_rois_dptr[3] * spatial_scale - align_offset;
    const T roi_end_h = offset_rois_dptr[4] * spatial_scale - align_offset;
    T roi_height = roi_end_h - roi_start_h;
    T roi_width = roi_end_w - roi_start_w;
    // aligned == false is for compatibility. the argument "aligned" doesn't have the semantic of
    // determining minimum roi size
    if (aligned == false) {
      roi_height = max(roi_height, static_cast<T>(1.0));
      roi_width = max(roi_width, static_cast<T>(1.0));
    }
    const T bin_height = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    const T bin_width = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    const int32_t bin_grid_height =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);
    const int32_t bin_grid_width =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = max(bin_grid_height * bin_grid_width, 1);
    const T* channel_dptr = in_dptr + (n * channel_num + c) * height * width;
    T out_val = 0.0;
    FOR_RANGE(int64_t, grid_i, 0, bin_grid_height) {
      // + .5f for center position
      T y = roi_start_h + h * bin_height
            + static_cast<T>(grid_i + 0.5f) * bin_height / static_cast<T>(bin_grid_height);
      FOR_RANGE(int64_t, grid_j, 0, bin_grid_width) {
        T x = roi_start_w + w * bin_width
              + static_cast<T>(grid_j + 0.5f) * bin_width / static_cast<T>(bin_grid_width);
        out_val += BilinearInterpolate(channel_dptr, height, width, y, x);
      }
    }
    out_dptr[index] = out_val / count;
  }
}

template<typename T>
__global__ void RoiAlignBackward(const int64_t nthreads, const T* out_diff_dptr, const T* rois_dptr,
                                 const T spatial_scale, const int32_t sampling_ratio,
                                 const int64_t channel_num, const int64_t height,
                                 const int64_t width, const int64_t pooled_height,
                                 const int64_t pooled_width, const bool aligned, T* in_diff_dptr) {
  const int64_t pooled_area = pooled_height * pooled_width;
  const int64_t channel_pooled_area = channel_num * pooled_height * pooled_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = index / channel_pooled_area;
    const T* offset_rois_dptr = rois_dptr + r * 5;
    const int64_t n = static_cast<int64_t>(offset_rois_dptr[0]);
    const T align_offset = aligned ? static_cast<T>(0.5) : static_cast<T>(0.f);
    const T roi_start_w = offset_rois_dptr[1] * spatial_scale - align_offset;
    const T roi_start_h = offset_rois_dptr[2] * spatial_scale - align_offset;
    const T roi_end_w = offset_rois_dptr[3] * spatial_scale - align_offset;
    const T roi_end_h = offset_rois_dptr[4] * spatial_scale - align_offset;
    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    // aligned == false is for compatibility. the argument "aligned" doesn't have the semantic of
    // determining minimum roi size
    if (aligned == false) {
      roi_height = max(roi_height, static_cast<T>(1.0));
      roi_width = max(roi_width, static_cast<T>(1.0));
    }
    const T bin_height = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    const T bin_width = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    const int32_t bin_grid_height =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);
    const int32_t bin_grid_width =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const T count = max(bin_grid_height * bin_grid_width, 1);
    const T bin_diff_avg = out_diff_dptr[index] / count;
    T* in_diff_channel_dptr = in_diff_dptr + (n * channel_num + c) * height * width;
    FOR_RANGE(int64_t, grid_i, 0, bin_grid_height) {
      // + .5f for center position
      T y = roi_start_h + h * bin_height
            + static_cast<T>(grid_i + 0.5f) * bin_height / static_cast<T>(bin_grid_height);
      FOR_RANGE(int64_t, grid_j, 0, bin_grid_width) {
        T x = roi_start_w + w * bin_width
              + static_cast<T>(grid_j + 0.5f) * bin_width / static_cast<T>(bin_grid_width);
        T diff11 = 0;
        T diff21 = 0;
        T diff12 = 0;
        T diff22 = 0;
        int32_t x_low = 0;
        int32_t x_high = 0;
        int32_t y_low = 0;
        int32_t y_high = 0;
        bool has_diff = BilinearInterpolateDiff(bin_diff_avg, height, width, y, x, diff11, diff21,
                                                diff12, diff22, x_low, x_high, y_low, y_high);
        if (has_diff) {
          const int64_t q11 = y_low * width + x_low;
          const int64_t q21 = y_low * width + x_high;
          const int64_t q12 = y_high * width + x_low;
          const int64_t q22 = y_high * width + x_high;
          atomicAdd(in_diff_channel_dptr + q11, diff11);
          atomicAdd(in_diff_channel_dptr + q21, diff21);
          atomicAdd(in_diff_channel_dptr + q12, diff12);
          atomicAdd(in_diff_channel_dptr + q22, diff22);
        }
      }
    }
  }
}

}  // namespace

template<typename T>
class RoIAlignKernel final : public user_op::OpKernel {
 public:
  RoIAlignKernel() = default;
  ~RoIAlignKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* rois_blob = ctx->Tensor4ArgNameAndIndex("rois", 0);
    if (rois_blob->shape_view().elem_cnt() == 0) { return; }
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t pooled_h = ctx->Attr<int32_t>("pooled_h");
    const int32_t pooled_w = ctx->Attr<int32_t>("pooled_w");
    const float spatial_scale = ctx->Attr<float>("spatial_scale");
    const int32_t sampling_ratio = ctx->Attr<int32_t>("sampling_ratio");
    const bool aligned = ctx->Attr<bool>("aligned");

    const int64_t elem_cnt = y_blob->shape_view().elem_cnt();
    RoiAlignForward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, x_blob->dptr<T>(), rois_blob->dptr<T>(), spatial_scale, sampling_ratio,
        x_blob->shape_view().At(1), x_blob->shape_view().At(2), x_blob->shape_view().At(3),
        pooled_h, pooled_w, aligned, y_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class RoIAlignGradKernel final : public user_op::OpKernel {
 public:
  RoIAlignGradKernel() = default;
  ~RoIAlignGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kCUDA>(ctx->stream(), dx_blob->mut_dptr<T>(), 0,
                              dx_blob->shape_view().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* rois_blob = ctx->Tensor4ArgNameAndIndex("rois", 0);
    const int32_t pooled_h = ctx->Attr<int32_t>("pooled_h");
    const int32_t pooled_w = ctx->Attr<int32_t>("pooled_w");
    const float spatial_scale = ctx->Attr<float>("spatial_scale");
    const int32_t sampling_ratio = ctx->Attr<int32_t>("sampling_ratio");
    const bool aligned = ctx->Attr<bool>("aligned");

    const int64_t elem_cnt = dy_blob->shape_view().elem_cnt();
    if (elem_cnt > 0) {
      RoiAlignBackward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, dy_blob->dptr<T>(), rois_blob->dptr<T>(), spatial_scale, sampling_ratio,
          dx_blob->shape_view().At(1), dx_blob->shape_view().At(2), dx_blob->shape_view().At(3),
          pooled_h, pooled_w, aligned, dx_blob->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("roi_align")
    .SetCreateFn<RoIAlignKernel<float>>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA);

REGISTER_USER_KERNEL("roi_align_grad")
    .SetCreateFn<RoIAlignGradKernel<float>>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA);

}  // namespace oneflow
