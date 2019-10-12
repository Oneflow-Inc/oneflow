#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__device__ T BilinearInterpolate(const T* channel_dptr, const int32_t height, const int32_t width,
                                 T y, T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) { return 0; }

  int32_t y_low = (y <= 0) ? 0 : y;
  int32_t x_low = (x <= 0) ? 0 : x;
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

  if (y > 0) { y_low = y; }
  if (x > 0) { x_low = x; }

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
__launch_bounds__(kCudaThreadsNumPerBlock) __global__
    void RoiAlignForward(const int64_t nthreads, const T* in_dptr, const T* rois_dptr,
                         const float spatial_scale, const int32_t sampling_ratio,
                         const int64_t channel_num, const int64_t height, const int64_t width,
                         const int64_t pooled_height, const int64_t pooled_width, T* out_dptr) {
  const int64_t pooled_area = pooled_height * pooled_width;
  const int64_t channel_pooled_area = channel_num * pooled_height * pooled_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = index / channel_pooled_area;
    const T* offset_rois_dptr = rois_dptr + r * 5;
    const int64_t n = static_cast<int64_t>(offset_rois_dptr[0]);
    const T roi_start_w = offset_rois_dptr[1] * spatial_scale;
    const T roi_start_h = offset_rois_dptr[2] * spatial_scale;
    const T roi_end_w = offset_rois_dptr[3] * spatial_scale;
    const T roi_end_h = offset_rois_dptr[4] * spatial_scale;
    const T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.f));
    const T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(1.f));
    const T bin_height = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    const T bin_width = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    const int32_t bin_grid_height =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);
    const int32_t bin_grid_width =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

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
    out_dptr[index] = out_val / (bin_grid_height * bin_grid_width);
  }
}

template<typename T>
__launch_bounds__(kCudaThreadsNumPerBlock) __global__
    void RoiAlignBackward(const int64_t nthreads, const T* out_diff_dptr, const T* rois_dptr,
                          const float spatial_scale, const int32_t sampling_ratio,
                          const int64_t channel_num, const int64_t height, const int64_t width,
                          const int64_t pooled_height, const int64_t pooled_width,
                          T* in_diff_dptr) {
  const int64_t pooled_area = pooled_height * pooled_width;
  const int64_t channel_pooled_area = channel_num * pooled_height * pooled_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = index / channel_pooled_area;
    const T* offset_rois_dptr = rois_dptr + r * 5;
    const int64_t n = static_cast<int64_t>(offset_rois_dptr[0]);
    const T roi_start_w = offset_rois_dptr[1] * spatial_scale;
    const T roi_start_h = offset_rois_dptr[2] * spatial_scale;
    const T roi_end_w = offset_rois_dptr[3] * spatial_scale;
    const T roi_end_h = offset_rois_dptr[4] * spatial_scale;
    const T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.0));
    const T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(1.0));
    const T bin_height = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    const T bin_width = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    const int32_t bin_grid_height =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);
    const int32_t bin_grid_width =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    T* in_diff_channel_dptr = in_diff_dptr + (n * channel_num + c) * height * width;
    const T bin_diff_avg = out_diff_dptr[index] / (bin_grid_height * bin_grid_width);
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
class RoiAlignGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoiAlignGPUKernel);
  RoiAlignGPUKernel() = default;
  ~RoiAlignGPUKernel() = default;

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const RoiAlignArgs& args = this->op_conf().roi_align_conf().roi_align_args();
    const Blob* x_blob = BnInOp2Blob("x");
    const Blob* rois_blob = BnInOp2Blob("rois");
    Blob* y_blob = BnInOp2Blob("y");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    RoiAlignForward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, x_blob->dptr<T>(), rois_blob->dptr<T>(), args.spatial_scale(),
        args.sampling_ratio(), x_blob->shape().At(1), x_blob->shape().At(2), x_blob->shape().At(3),
        args.pooled_h(), args.pooled_w(), y_blob->mut_dptr<T>());
  }
};

template<typename T>
class RoiAlignGradGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoiAlignGradGPUKernel);
  RoiAlignGradGPUKernel() = default;
  ~RoiAlignGradGPUKernel() = default;

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* dx_blob = BnInOp2Blob("dx");
    if (dx_blob == nullptr) { return; }
    const RoiAlignArgs& args = this->op_conf().roi_align_grad_conf().roi_align_args();
    Memset<DeviceType::kGPU>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                             dx_blob->ByteSizeOfBlobBody());
    const Blob* dy_blob = BnInOp2Blob("dy");
    const Blob* rois_blob = BnInOp2Blob("rois");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    RoiAlignBackward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, dy_blob->dptr<T>(), rois_blob->dptr<T>(), args.spatial_scale(),
        args.sampling_ratio(), dx_blob->shape().At(1), dx_blob->shape().At(2),
        dx_blob->shape().At(3), args.pooled_h(), args.pooled_w(), dx_blob->mut_dptr<T>());
  }
};

#define REGISTER_ROIALIGN_GPU_KERNEL(dtype)                                                       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kRoiAlignConf, DeviceType::kGPU, dtype,     \
                                        RoiAlignGPUKernel<dtype>);                                \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kRoiAlignGradConf, DeviceType::kGPU, dtype, \
                                        RoiAlignGradGPUKernel<dtype>)

REGISTER_ROIALIGN_GPU_KERNEL(float);
REGISTER_ROIALIGN_GPU_KERNEL(double);

}  // namespace oneflow
