#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T, typename K>
__device__ T BilinearInterpolate(const K* input, const int height, const int width, T y, T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) { return 0; }

  int y_low = (y <= 0) ? 0 : y;
  int x_low = (x <= 0) ? 0 : x;
  int y_high = 0;
  int x_high = 0;

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
  const T hy = GetOneVal<T>() - ly;
  const T hx = GetOneVal<T>() - lx;

  // https://en.wikipedia.org/wiki/Bilinear_interpolation
  const int q11 = y_low * width + x_low;
  const int q21 = y_low * width + x_high;
  const int q12 = y_high * width + x_low;
  const int q22 = y_high * width + x_high;
  // no 1 / (x_high - x_low) * (y_high - y_low) because it will always be 1
  return hy * hx * static_cast<T>(input[q11]) + hy * lx * static_cast<T>(input[q21])
         + ly * hx * static_cast<T>(input[q12]) + ly * lx * static_cast<T>(input[q22]);
}

template<typename T>
__launch_bounds__(kCudaThreadsNumPerBlock) __global__
    void MasksCropAndResizeForward(const int nthreads, const int8_t* masks, const T* rois,
                                   const int channels, const int height, const int width,
                                   const int mask_height, const int mask_width, T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int n = index / (mask_height * mask_width * channels);
    const int c = (index / (mask_height * mask_width)) % channels;
    const int h = (index / mask_width) % mask_height;
    const int w = index % mask_width;

    const T roi_x_min = rois[n * 4 + 0];
    const T roi_y_min = rois[n * 4 + 1];
    const T roi_x_max = rois[n * 4 + 2];
    const T roi_y_max = rois[n * 4 + 3];

    const T roi_height = max(roi_y_max - roi_y_min, GetOneVal<T>());
    const T roi_width = max(roi_x_max - roi_x_min, GetOneVal<T>());
    const T bin_height = static_cast<T>(roi_height) / static_cast<T>(mask_height);
    const T bin_width = static_cast<T>(roi_width) / static_cast<T>(mask_width);
    const int bin_grid_height = ceil(roi_height / mask_height);
    const int bin_grid_width = ceil(roi_width / mask_width);

    const int8_t* cur_mask = masks + (n * channels + c) * height * width;
    T out_val = static_cast<T>(0.0f);
    FOR_RANGE(int64_t, grid_i, 0, bin_grid_height) {
      // + .5f for center position
      const T y = roi_y_min + h * bin_height
                  + static_cast<T>(grid_i + 0.5f) * bin_height / static_cast<T>(bin_grid_height);
      FOR_RANGE(int64_t, grid_j, 0, bin_grid_width) {
        const T x = roi_x_min + w * bin_width
                    + static_cast<T>(grid_j + 0.5f) * bin_width / static_cast<T>(bin_grid_width);
        out_val += BilinearInterpolate(cur_mask, height, width, y, x);
      }
    }
    output[index] = out_val;
  }
}

}  // namespace

template<typename T>
class MasksCropAndResizeGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MasksCropAndResizeGPUKernel);
  MasksCropAndResizeGPUKernel() = default;
  ~MasksCropAndResizeGPUKernel() = default;

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const auto& conf = this->op_conf().masks_crop_and_resize_conf();
    const Blob* masks_blob = BnInOp2Blob("masks");
    const Blob* rois_blob = BnInOp2Blob("rois");
    Blob* out_blob = BnInOp2Blob("out");
    const int elem_cnt = out_blob->shape().elem_cnt();
    const int channels = masks_blob->shape().At(1);
    const int height = masks_blob->shape().At(2);
    const int width = masks_blob->shape().At(3);
    MasksCropAndResizeForward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                   ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, masks_blob->dptr<int8_t>(), rois_blob->dptr<T>(), channels, height, width,
        conf.mask_height(), conf.mask_width(), out_blob->mut_dptr<T>());
  }
};

#define REGISTER_MASKS_CROP_AND_RESIZE_GPU_KERNEL(dtype)                                         \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kMasksCropAndResizeConf, DeviceType::kGPU, \
                                        dtype, MasksCropAndResizeGPUKernel<dtype>);

REGISTER_MASKS_CROP_AND_RESIZE_GPU_KERNEL(float);
REGISTER_MASKS_CROP_AND_RESIZE_GPU_KERNEL(double);

}  // namespace oneflow
