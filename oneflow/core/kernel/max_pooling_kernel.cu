#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/max_pooling_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void MaxPoolForward(const int64_t nthreads, const T* in_dptr,
                               T* out_dptr, uint32_t* mask_dptr,
                               const int64_t channels, const int64_t height,
                               const int64_t width, const int64_t pooled_height,
                               const int64_t pooled_width, PoolingCudaCtx ctx) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t pw = index % pooled_width;
    const int64_t ph = (index / pooled_width) % pooled_height;
    const int64_t c = (index / pooled_width / pooled_height) % channels;
    const int64_t n = index / pooled_width / pooled_height / channels;
    int64_t hstart = ph * ctx.strides_h - ctx.padding_top;
    int64_t wstart = pw * ctx.strides_w - ctx.padding_left;
    const int64_t hend = (hstart + ctx.pool_size_h < height)
                             ? (hstart + ctx.pool_size_h)
                             : height;
    const int64_t wend =
        (wstart + ctx.pool_size_w < width) ? (wstart + ctx.pool_size_w) : width;
    hstart = (hstart > 0) ? hstart : 0;
    wstart = (wstart > 0) ? wstart : 0;
    const T* const in_slice = in_dptr + (n * channels + c) * height * width;
    T maxval = in_slice[hstart * width + wstart];
    uint32_t maxidx = hstart * width + wstart;
    for (int64_t h = hstart; h < hend; ++h) {
      for (int64_t w = wstart; w < wend; ++w) {
        if (in_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = in_slice[maxidx];
        }
      }
    }
    out_dptr[index] = maxval;
    mask_dptr[index] = maxidx;
  }
}

template<typename T>
__global__ void MaxPoolBackward(const int64_t nthreads, const T* out_diff_dptr,
                                const uint32_t* mask_dptr, T* in_diff_dptr,
                                const int64_t channels, const int64_t height,
                                const int64_t width,
                                const int64_t pooled_height,
                                const int64_t pooled_width,
                                PoolingCudaCtx ctx) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t w = index % width;
    const int64_t h = (index / width) % height;
    const int64_t c = (index / width / height) % channels;
    const int64_t n = index / width / height / channels;
    int64_t phstart =
        (h + ctx.padding_top < ctx.pool_size_h)
            ? 0
            : (h + ctx.padding_top - ctx.pool_size_h) / ctx.strides_h + 1;
    int64_t pwstart =
        (w + ctx.padding_left < ctx.pool_size_w)
            ? 0
            : (w + ctx.padding_left - ctx.pool_size_w) / ctx.strides_w + 1;
    const int64_t phend =
        ((h + ctx.padding_bottom) / ctx.strides_h + 1 < pooled_height)
            ? ((h + ctx.padding_bottom) / ctx.strides_h + 1)
            : pooled_height;
    const int64_t pwend =
        ((w + ctx.padding_right) / ctx.strides_w + 1 < pooled_width)
            ? ((w + ctx.padding_right) / ctx.strides_w + 1)
            : pooled_width;
    T gradient = 0;
    const int64_t offset = (n * channels + c) * pooled_height * pooled_width;
    const T* const out_diff_slice = out_diff_dptr + offset;
    const uint32_t* const mask_slice = mask_dptr + offset;
    for (int64_t ph = phstart; ph < phend; ++ph) {
      for (int64_t pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += out_diff_slice[ph * pooled_width + pw];
        }
      }
    }
    in_diff_dptr[index] = gradient;
  }
}

}  // namespace

template<typename T>
class MaxPoolingKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingKernelUtil);
  MaxPoolingKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const Blob* in_blob, Blob* out_blob,
                      Blob* mask_blob, const MaxPoolingOpConf& op_conf,
                      const PoolingKernelConf& kernel_conf) {
    const int64_t count = out_blob->shape().elem_cnt();
    PoolingCudaCtx pooling_cuda_ctx = BuildPoolingCudaCtx(op_conf, kernel_conf);
    MaxPoolForward<T><<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(
        count, in_blob->dptr<T>(), out_blob->mut_dptr<T>(),
        mask_blob->mut_dptr<uint32_t>(), in_blob->shape().At(1),
        in_blob->shape().At(2), in_blob->shape().At(3), out_blob->shape().At(2),
        out_blob->shape().At(3), pooling_cuda_ctx);
  }

  static void Backward(const KernelCtx& ctx, const Blob* out_diff_blob,
                       const Blob* mask_blob, Blob* in_diff_blob,
                       const MaxPoolingOpConf& op_conf,
                       const PoolingKernelConf& kernel_conf) {
    const int64_t count = in_diff_blob->shape().elem_cnt();
    PoolingCudaCtx pooling_cuda_ctx = BuildPoolingCudaCtx(op_conf, kernel_conf);
    MaxPoolBackward<T><<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock,
                         0, ctx.device_ctx->cuda_stream()>>>(
        count, out_diff_blob->dptr<T>(), mask_blob->dptr<uint32_t>(),
        in_diff_blob->mut_dptr<T>(), in_diff_blob->shape().At(1),
        in_diff_blob->shape().At(2), in_diff_blob->shape().At(3),
        out_diff_blob->shape().At(2), out_diff_blob->shape().At(3),
        pooling_cuda_ctx);
  }
};

#define INSTANTIATE_MAX_POOLING_KERNEL_UTIL(type_cpp, type_proto) \
  template class MaxPoolingKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_MAX_POOLING_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
