#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__device__ __forceinline__ T Round(T x);

template<>
__device__ __forceinline__ float Round(float x) {
  return rintf(x);
}

template<>
__device__ __forceinline__ double Round(double x) {
  return rint(x);
}

template<typename T>
__device__ __forceinline__ T Floor(T x);

template<>
__device__ __forceinline__ float Floor(float x) {
  return floorf(x);
}

template<>
__device__ __forceinline__ double Floor(double x) {
  return floor(x);
}

template<typename T>
__launch_bounds__(kCudaThreadsNumPerBlock) __global__
    void MasksCropAndResizeForward(const int elem_cnt, const int8_t* masks, const T* rois,
                                   const int channels, const int height, const int width,
                                   const int mask_height, const int mask_width, T* output) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    const int n = index / (mask_height * mask_width * channels);
    const int c = (index / (mask_height * mask_width)) % channels;
    const int h = (index / mask_width) % mask_height;
    const int w = index % mask_width;

    const T roi_x_min = Round(rois[n * 4 + 0]);
    const T roi_y_min = Round(rois[n * 4 + 1]);
    const T roi_x_max = Round(rois[n * 4 + 2]);
    const T roi_y_max = Round(rois[n * 4 + 3]);
    assert(roi_x_min >= 0);
    assert(roi_y_min >= 0);
    assert(roi_x_max <= width - 1);
    assert(roi_y_max <= height - 1);
    assert(roi_x_min <= roi_x_max - 1);
    assert(roi_y_min <= roi_y_max - 1);
    const T roi_height = roi_y_max - roi_y_min;
    const T roi_width = roi_x_max - roi_x_min;
    const T bin_height = roi_height / static_cast<T>(mask_height);
    const T bin_width = roi_width / static_cast<T>(mask_width);

    const int8_t* cur_mask = masks + (n * channels + c) * height * width;
    const T x_center =
        max(roi_x_min + static_cast<T>((w + 0.5f) * bin_width - 0.5f), GetZeroVal<T>());
    const T y_center =
        max(roi_y_min + static_cast<T>((h + 0.5f) * bin_height - 0.5f), GetZeroVal<T>());
    assert(x_center >= 0);
    assert(y_center >= 0);
    assert(x_center < width);
    assert(y_center < height);

    const int x_low = static_cast<int>(Floor(x_center));
    const int y_low = static_cast<int>(Floor(y_center));
    const int x_high = x_low < width - 1 ? x_low + 1 : x_low;
    const int y_high = y_low < height - 1 ? y_low + 1 : y_low;
    const T x_w2 = x_center - static_cast<T>(x_low);
    const T y_w2 = y_center - static_cast<T>(y_low);
    const T x_w1 = GetOneVal<T>() - x_w2;
    const T y_w1 = GetOneVal<T>() - y_w2;
    const int n11 = y_low * width + x_low;
    const int n12 = y_low * width + x_high;
    const int n21 = y_high * width + x_low;
    const int n22 = y_high * width + x_high;
    output[index] =
        y_w1 * (x_w1 * static_cast<T>(cur_mask[n11]) + x_w2 * static_cast<T>(cur_mask[n12]))
        + y_w2 * (x_w1 * static_cast<T>(cur_mask[n21]) + x_w2 * static_cast<T>(cur_mask[n22]));
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
