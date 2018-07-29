#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/roi_align_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__device__ T BilinearInterpolate(const T* in_dptr, const int height, const int width, T y, T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) { return 0; }

  if (y <= 0) { y = 0; }
  if (x <= 0) { x = 0; }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  T v1 = in_dptr[y_low * width + x_low];
  T v2 = in_dptr[y_low * width + x_high];
  T v3 = in_dptr[y_high * width + x_low];
  T v4 = in_dptr[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template<typename T>
__global__ void RoIAlignForward(const int64_t nthreads, const T* in_dptr, const float spatial_scale,
                                const int32_t sampling_ratio, const int64_t channel_num,
                                const int64_t height, const int64_t width, const int64_t roi_num,
                                const int64_t pooled_height, const int64_t pooled_width,
                                const T* rois_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t pooled_area = pooled_width * pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = (index / pooled_area / channel_num) % roi_num;
    const int64_t n = index / pooled_area / channel_num / roi_num;
    const T* offset_rois = rois_dptr + (n * roi_num + r) * 4;

    const T roi_start_w = offset_rois[0] * spatial_scale;
    const T roi_start_h = offset_rois[1] * spatial_scale;
    const T roi_end_w = offset_rois[2] * spatial_scale;
    const T roi_end_h = offset_rois[3] * spatial_scale;

    // not rounded, no need to +1
    const T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    const T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    const T bin_height = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    const T bin_width = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    const T* offset_in_dptr = in_dptr + (n * channel_num + c) * height * width;

    // adaptive if sampling ratio is negative
    // height and width here are not pixel count but grid size
    const int32_t bin_grid_height =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);
    const int32_t bin_grid_width =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T bin_grid_density_h = bin_height / static_cast<T>(bin_grid_height);
    const T bin_grid_density_w = bin_width / static_cast<T>(bin_grid_width);

    const T count = bin_grid_height * bin_grid_width;
    T out_val = 0.;
    FOR_RANGE(int64_t, grid_h, 0, bin_grid_height) {
      {
        // + .5f for center position
        const T y =
            roi_start_h + h * bin_height + static_cast<T>(grid_h + .5f) * bin_grid_density_h;
        FOR_RANGE(int64_t, grid_w, 0, bin_grid_width) {
          const T x =
              roi_start_w + w * bin_width + static_cast<T>(grid_w + .5f) * bin_grid_density_w;
          // pass height and width in case out of the boundary of feat. map
          T val = BilinearInterpolate(offset_in_dptr, height, width, y, x);
          out_val += val;
        }
      }
    }
    // average pooling
    out_val /= count;
    out_dptr[index] = out_val;
  }
}
}  // namespace

template<typename T>
class RoIAlignKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIAlignKernelUtil);
  RoIAlignKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob) {
    const int64_t count = out_blob->shape().elem_cnt();
    RoIAlignForward<T><<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(
        count, in_blob->dptr<T>(), conf.spatial_scale(), conf.sampling_ratio(),
        in_blob->shape().At(1), in_blob->shape().At(2), in_blob->shape().At(3),
        rois_blob->shape().At(1), conf.pooled_h(), conf.pooled_w(), rois_blob->dptr<T>(),
        out_blob->mut_dptr<T>());
  }

  static void Backward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* out_diff_blob,
                       const Blob* rois_blob, Blob* in_diff_blob) {}
};

#define INSTANTIATE_ROI_ALIGN_KERNEL_UTIL(type_cpp, type_proto) \
  template class RoIAlignKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_ROI_ALIGN_KERNEL_UTIL,
                     OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat))

}  // namespace oneflow
