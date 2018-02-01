#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/average_pooling_2d_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void AveragePooling2DForward(
    const int64_t nthreads, const T* in_dptr, T* out_dptr,
    const int64_t channels, const int64_t height, const int64_t width,
    const int64_t pooled_height, const int64_t pooled_width,
    const Pooling2DCtx ctx) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t pw = index % pooled_width;
    const int64_t ph = (index / pooled_width) % pooled_height;
    const int64_t c = (index / pooled_width / pooled_height) % channels;
    const int64_t n = index / pooled_width / pooled_height / channels;
    int64_t hstart = ph * ctx.strides_h - ctx.padding_top;
    int64_t wstart = pw * ctx.strides_w - ctx.padding_left;
    int64_t hend = (hstart + ctx.pool_size_h < height + ctx.padding_bottom)
                       ? (hstart + ctx.pool_size_h)
                       : (height + ctx.padding_bottom);
    int64_t wend = (wstart + ctx.pool_size_w < width + ctx.padding_right)
                       ? (wstart + ctx.pool_size_w)
                       : (width + ctx.padding_right);
    const int64_t pool_size = (hend - hstart) * (wend - wstart);
    hstart = (hstart > 0) ? hstart : 0;
    wstart = (wstart > 0) ? wstart : 0;
    hend = (hend < height) ? hend : height;
    wend = (wend < width) ? wend : width;
    T aveval = 0;
    const T* const in_slice = in_dptr + (n * channels + c) * height * width;
    for (int64_t h = hstart; h < hend; ++h) {
      for (int64_t w = wstart; w < wend; ++w) {
        aveval += in_slice[h * width + w];
      }
    }
    out_dptr[index] = aveval / pool_size;
  }
}

template<typename T>
__global__ void AveragePooling2DBackward(
    const int64_t nthreads, const T* out_diff_dptr, T* in_diff_dptr,
    const int64_t channels, const int64_t height, const int64_t width,
    const int64_t pooled_height, const int64_t pooled_width, Pooling2DCtx ctx) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t w = index % width + ctx.padding_left;
    const int64_t h = (index / width) % height + ctx.padding_top;
    const int64_t c = (index / width / height) % channels;
    const int64_t n = index / width / height / channels;
    int64_t phstart =
        (h < ctx.pool_size_h) ? 0 : (h - ctx.pool_size_h) / ctx.strides_h + 1;
    int64_t pwstart =
        (w < ctx.pool_size_w) ? 0 : (w - ctx.pool_size_w) / ctx.strides_w + 1;
    const int64_t phend = (h / ctx.strides_h + 1 < pooled_height)
                              ? (h / ctx.strides_h + 1)
                              : pooled_height;
    const int64_t pwend = (w / ctx.strides_w + 1 < pooled_width)
                              ? (w / ctx.strides_w + 1)
                              : pooled_width;
    T gradient = 0;
    const int64_t offset = (n * channels + c) * pooled_height * pooled_width;
    const T* const out_diff_slice = out_diff_dptr + offset;
    for (int64_t ph = phstart; ph < phend; ++ph) {
      for (int64_t pw = pwstart; pw < pwend; ++pw) {
        int64_t hstart = ph * ctx.strides_h - ctx.padding_top;
        int64_t wstart = pw * ctx.strides_w - ctx.padding_left;
        int64_t hend = (hstart + ctx.pool_size_h < height + ctx.padding_bottom)
                           ? (hstart + ctx.pool_size_h)
                           : (height + ctx.padding_bottom);
        int64_t wend = (wstart + ctx.pool_size_w < width + ctx.padding_right)
                           ? (wstart + ctx.pool_size_w)
                           : (width + ctx.padding_right);
        int64_t pool_size = (hend - hstart) * (wend - wstart);
        gradient += out_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    in_diff_dptr[index] = gradient;
  }
}

}  // namespace

template<typename T>
class AveragePooling2DKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling2DKernelUtil);
  AveragePooling2DKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const Blob* in_blob, Blob* out_blob,
                      const Pooling2DCtx& pooling_ctx) {
    const int64_t count = out_blob->shape().elem_cnt();
    Pooling2DCtx cuda_ctx = pooling_ctx;
    AveragePooling2DForward<T>
        <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            count, in_blob->dptr<T>(), out_blob->mut_dptr<T>(),
            in_blob->shape().At(1), in_blob->shape().At(2),
            in_blob->shape().At(3), out_blob->shape().At(2),
            out_blob->shape().At(3), cuda_ctx);
  }

  static void Backward(const KernelCtx& ctx, const Blob* out_diff_blob,
                       Blob* in_diff_blob, const Pooling2DCtx& pooling_ctx) {
    const int64_t count = in_diff_blob->shape().elem_cnt();
    Pooling2DCtx cuda_ctx = pooling_ctx;
    AveragePooling2DBackward<T>
        <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            count, out_diff_blob->dptr<T>(), in_diff_blob->mut_dptr<T>(),
            in_diff_blob->shape().At(1), in_diff_blob->shape().At(2),
            in_diff_blob->shape().At(3), out_diff_blob->shape().At(2),
            out_diff_blob->shape().At(3), cuda_ctx);
  }
};

#define INSTANTIATE_AVERAGE_POOLING_2D_KERNEL_UTIL(type_cpp, type_proto) \
  template class AveragePooling2DKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_AVERAGE_POOLING_2D_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
