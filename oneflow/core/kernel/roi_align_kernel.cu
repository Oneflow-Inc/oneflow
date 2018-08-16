#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/roi_align_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__device__ T BilinearInterpolate(const T* channel_dptr, const int32_t height, const int32_t width,
                                 const T y, const T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) { return 0; }

  const int32_t y_low = (y <= 0) ? 0 : y;
  const int32_t x_low = (x <= 0) ? 0 : x;

  if (y_low >= height - 1 || x_low >= width - 1) { return 0; }
  const int32_t y_high = y_low + 1;
  const int32_t x_high = x_low + 1;

  const T ly = y - y_low;
  const T lx = x - x_low;
  const T hy = y_high - y;
  const T hx = x_high - x;

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
                                        const int64_t width, const T y, const T x, T& diff11,
                                        T& diff21, T& diff12, T& diff22, int32_t& x_low,
                                        int32_t& x_high, int32_t& y_low, int32_t& y_high) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) { return false; }

  if (y > 0) { y_low = y; }
  if (x > 0) { x_low = x; }

  if (y_low >= height - 1 || x_low >= width - 1) { return false; }
  y_high = y_low + 1;
  x_high = x_low + 1;

  const T ly = y - y_low;
  const T lx = x - x_low;
  const T hy = y_high - y;
  const T hx = x_high - x;

  diff11 = bin_diff_avg * hy * hx;
  diff21 = bin_diff_avg * hy * lx;
  diff12 = bin_diff_avg * ly * hx;
  diff22 = bin_diff_avg * ly * lx;
  return true;
}

template<typename T>
__global__ void RoIAlignForward(const int64_t nthreads, const T* in_dptr, const float spatial_scale,
                                const int32_t sampling_ratio, const int64_t channel_num,
                                const int64_t height, const int64_t width, const int64_t roi_num,
                                const int64_t pooled_height, const int64_t pooled_width,
                                const T* rois_dptr, T* out_dptr) {
  const int64_t pooled_area = pooled_height * pooled_width;
  const int64_t channel_pooled_area = channel_num * pooled_height * pooled_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = (index / channel_pooled_area) % roi_num;
    const int64_t n = index / channel_pooled_area / roi_num;
    const T* offset_rois_dptr = rois_dptr + (n * roi_num + r) * 4;
    const T roi_start_w = offset_rois_dptr[0] * spatial_scale;
    const T roi_start_h = offset_rois_dptr[1] * spatial_scale;
    const T roi_end_w = offset_rois_dptr[2] * spatial_scale;
    const T roi_end_h = offset_rois_dptr[3] * spatial_scale;
    const T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.0));
    const T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(1.0));
    const T bin_height = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    const T bin_width = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    const int32_t bin_grid_height =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);
    const int32_t bin_grid_width =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T bin_grid_density_h = bin_height / static_cast<T>(bin_grid_height);
    const T bin_grid_density_w = bin_width / static_cast<T>(bin_grid_width);

    const T* channel_dptr = in_dptr + (n * channel_num + c) * height * width;
    T out_val = 0.0;
    FOR_RANGE(int64_t, grid_i, 0, bin_grid_height) {
      // + .5f for center position
      const T y = roi_start_h + h * bin_height + static_cast<T>(grid_i + 0.5f) * bin_grid_density_h;
      FOR_RANGE(int64_t, grid_j, 0, bin_grid_width) {
        const T x =
            roi_start_w + w * bin_width + static_cast<T>(grid_j + 0.5f) * bin_grid_density_w;
        out_val += BilinearInterpolate(channel_dptr, height, width, y, x);
      }
    }
    // average pooling
    out_dptr[index] = out_val / (bin_grid_height * bin_grid_width);
  }
}

template<typename T>
__global__ void RoIAlignBackward(const int64_t nthreads, const T* out_diff_dptr,
                                 const float spatial_scale, const int32_t sampling_ratio,
                                 const int64_t channel_num, const int64_t height,
                                 const int64_t width, const int64_t roi_num,
                                 const int64_t pooled_height, const int64_t pooled_width,
                                 const T* rois_dptr, T* in_diff_dptr) {
  const int64_t pooled_area = pooled_height * pooled_width;
  const int64_t channel_pooled_area = channel_num * pooled_height * pooled_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = (index / channel_pooled_area) % roi_num;
    const int64_t n = index / channel_pooled_area / roi_num;
    const T* offset_rois_dptr = rois_dptr + (n * roi_num + r) * 4;
    const T roi_start_w = offset_rois_dptr[0] * spatial_scale;
    const T roi_start_h = offset_rois_dptr[1] * spatial_scale;
    const T roi_end_w = offset_rois_dptr[2] * spatial_scale;
    const T roi_end_h = offset_rois_dptr[3] * spatial_scale;
    const T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.0));
    const T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(1.0));
    const T bin_height = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    const T bin_width = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    const int32_t bin_grid_height =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);
    const int32_t bin_grid_width =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T bin_grid_density_h = bin_height / static_cast<T>(bin_grid_height);
    const T bin_grid_density_w = bin_width / static_cast<T>(bin_grid_width);

    T* in_diff_channel_dptr = in_diff_dptr + (n * channel_num + c) * height * width;
    const T* out_diff_channel_dptr = out_diff_dptr + (n * channel_num + c) * pooled_area;
    const T bin_diff = out_diff_channel_dptr[h * pooled_width + w];
    FOR_RANGE(int64_t, grid_i, 0, bin_grid_height) {
      // + .5f for center position
      const T y = roi_start_h + h * bin_height + static_cast<T>(grid_i + 0.5f) * bin_grid_density_h;
      FOR_RANGE(int64_t, grid_j, 0, bin_grid_width) {
        const T x =
            roi_start_w + w * bin_width + static_cast<T>(grid_j + 0.5f) * bin_grid_density_w;
        T diff11 = 0;
        T diff21 = 0;
        T diff12 = 0;
        T diff22 = 0;
        int32_t x_low = 0;
        int32_t x_high = 0;
        int32_t y_low = 0;
        int32_t y_high = 0;
        bool has_diff = BilinearInterpolateDiff(bin_diff / (bin_grid_height * bin_grid_width),
                                                height, width, y, x, diff11, diff21, diff12, diff22,
                                                x_low, x_high, y_low, y_high);
        if (has_diff) {
          const int64_t q11 = y_low * width + x_low;
          const int64_t q21 = y_low * width + x_high;
          const int64_t q12 = y_high * width + x_low;
          const int64_t q22 = y_high * width + x_high;
          gpu_atomic_add(in_diff_channel_dptr + q11, diff11);
          gpu_atomic_add(in_diff_channel_dptr + q21, diff21);
          gpu_atomic_add(in_diff_channel_dptr + q12, diff12);
          gpu_atomic_add(in_diff_channel_dptr + q22, diff22);
        }
      }
    }
  }
}
}  // namespace

template<typename T>
struct RoIAlignKernelUtil<DeviceType::kGPU, T> {
  static void Forward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob) {
    const int64_t elem_cnt = out_blob->shape().elem_cnt();
    RoIAlignForward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, in_blob->dptr<T>(), conf.spatial_scale(), conf.sampling_ratio(),
        in_blob->shape().At(1), in_blob->shape().At(2), in_blob->shape().At(3),
        rois_blob->shape().At(1), conf.pooled_h(), conf.pooled_w(), rois_blob->dptr<T>(),
        out_blob->mut_dptr<T>());
  }

  static void Backward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* out_diff_blob,
                       const Blob* rois_blob, Blob* in_diff_blob) {
    const int64_t elem_cnt = out_diff_blob->shape().elem_cnt();
    RoIAlignBackward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, out_diff_blob->dptr<T>(), conf.spatial_scale(), conf.sampling_ratio(),
        in_diff_blob->shape().At(1), in_diff_blob->shape().At(2), in_diff_blob->shape().At(3),
        rois_blob->shape().At(1), conf.pooled_h(), conf.pooled_w(), rois_blob->dptr<T>(),
        in_diff_blob->mut_dptr<T>());
  }
};

#define INSTANTIATE_ROI_ALIGN_KERNEL_UTIL(type_cpp, type_proto) \
  template class RoIAlignKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_ROI_ALIGN_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
