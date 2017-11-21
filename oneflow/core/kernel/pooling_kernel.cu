#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void MaxPoolForward(const int64_t nthreads, const T* in_dptr,
                               T* out_dptr, uint32_t* mask_dptr,
                               const int64_t channels, const int64_t height,
                               const int64_t width, const int64_t pooled_height,
                               const int64_t pooled_width,
                               const int64_t kernel_h, const int64_t kernel_w,
                               const int64_t stride_h, const int64_t stride_w,
                               const int64_t pad_h, const int64_t pad_w) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t pw = index % pooled_width;
    const int64_t ph = (index / pooled_width) % pooled_height;
    const int64_t c = (index / pooled_width / pooled_height) % channels;
    const int64_t n = index / pooled_width / pooled_height / channels;
    int64_t hstart = ph * stride_h - pad_h;
    int64_t wstart = pw * stride_w - pad_w;
    const int64_t hend =
        (hstart + kernel_h < height) ? (hstart + kernel_h) : height;
    const int64_t wend =
        (wstart + kernel_w < width) ? (wstart + kernel_w) : width;
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
__global__ void AvePoolForward(const int64_t nthreads, const T* in_dptr,
                               T* out_dptr, const int64_t channels,
                               const int64_t height, const int64_t width,
                               const int64_t pooled_height,
                               const int64_t pooled_width,
                               const int64_t kernel_h, const int64_t kernel_w,
                               const int64_t stride_h, const int64_t stride_w,
                               const int64_t pad_h, const int64_t pad_w) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t pw = index % pooled_width;
    const int64_t ph = (index / pooled_width) % pooled_height;
    const int64_t c = (index / pooled_width / pooled_height) % channels;
    const int64_t n = index / pooled_width / pooled_height / channels;
    int64_t hstart = ph * stride_h - pad_h;
    int64_t wstart = pw * stride_w - pad_w;
    int64_t hend = (hstart + kernel_h < height + pad_h) ? (hstart + kernel_h)
                                                        : (height + pad_h);
    int64_t wend = (wstart + kernel_w < width + pad_w) ? (wstart + kernel_w)
                                                       : (width + pad_w);
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
__global__ void MaxPoolBackward(const int64_t nthreads, const T* out_diff_dptr,
                                const uint32_t* mask_dptr, T* in_diff_dptr,
                                const int64_t channels, const int64_t height,
                                const int64_t width,
                                const int64_t pooled_height,
                                const int64_t pooled_width,
                                const int64_t kernel_h, const int64_t kernel_w,
                                const int64_t stride_h, const int64_t stride_w,
                                const int64_t pad_h, const int64_t pad_w) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t w = index % width;
    const int64_t h = (index / width) % height;
    const int64_t c = (index / width / height) % channels;
    const int64_t n = index / width / height / channels;
    int64_t phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int64_t pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int64_t phend = ((h + pad_h) / stride_h + 1 < pooled_height)
                              ? ((h + pad_h) / stride_h + 1)
                              : pooled_height;
    const int64_t pwend = ((w + pad_w) / stride_w + 1 < pooled_width)
                              ? ((w + pad_w) / stride_w + 1)
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

template<typename T>
__global__ void AvePoolBackward(const int64_t nthreads, const T* out_diff_dptr,
                                T* in_diff_dptr, const int64_t channels,
                                const int64_t height, const int64_t width,
                                const int64_t pooled_height,
                                const int64_t pooled_width,
                                const int64_t kernel_h, const int64_t kernel_w,
                                const int64_t stride_h, const int64_t stride_w,
                                const int64_t pad_h, const int64_t pad_w) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t w = index % width + pad_w;
    const int64_t h = (index / width) % height + pad_h;
    const int64_t c = (index / width / height) % channels;
    const int64_t n = index / width / height / channels;
    int64_t phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int64_t pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int64_t phend =
        (h / stride_h + 1 < pooled_height) ? (h / stride_h + 1) : pooled_height;
    const int64_t pwend =
        (w / stride_w + 1 < pooled_width) ? (w / stride_w + 1) : pooled_width;
    T gradient = 0;
    const int64_t offset = (n * channels + c) * pooled_height * pooled_width;
    const T* const out_diff_slice = out_diff_dptr + offset;
    for (int64_t ph = phstart; ph < phend; ++ph) {
      for (int64_t pw = pwstart; pw < pwend; ++pw) {
        int64_t hstart = ph * stride_h - pad_h;
        int64_t wstart = pw * stride_w - pad_w;
        int64_t hend = (hstart + kernel_h < height + pad_h)
                           ? (hstart + kernel_h)
                           : (height + pad_h);
        int64_t wend = (wstart + kernel_w < width + pad_w) ? (wstart + kernel_w)
                                                           : (width + pad_w);
        int64_t pool_size = (hend - hstart) * (wend - wstart);
        gradient += out_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    in_diff_dptr[index] = gradient;
  }
}

}  // namespace

template<typename T>
class PoolingKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx& ctx, const Blob* in_blob,
                             Blob* out_blob, Blob* mask_blob,
                             const PoolingOpConf& pooling_conf) {
    const int64_t count = out_blob->shape().elem_cnt();

    switch (pooling_conf.pool()) {
      case PoolingOpConf::kMax: {
        MaxPoolForward<T>
            <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
               ctx.device_ctx->cuda_stream()>>>(
                count, in_blob->dptr<T>(), out_blob->mut_dptr<T>(),
                mask_blob->mut_dptr<uint32_t>(), in_blob->shape().At(1),
                in_blob->shape().At(2), in_blob->shape().At(3),
                out_blob->shape().At(2), out_blob->shape().At(3),
                pooling_conf.kernel_h(), pooling_conf.kernel_w(),
                pooling_conf.stride_h(), pooling_conf.stride_w(),
                pooling_conf.pad_h(), pooling_conf.pad_w());
        break;
      }
      case PoolingOpConf::kAve: {
        AvePoolForward<T>
            <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
               ctx.device_ctx->cuda_stream()>>>(
                count, in_blob->dptr<T>(), out_blob->mut_dptr<T>(),
                in_blob->shape().At(1), in_blob->shape().At(2),
                in_blob->shape().At(3), out_blob->shape().At(2),
                out_blob->shape().At(3), pooling_conf.kernel_h(),
                pooling_conf.kernel_w(), pooling_conf.stride_h(),
                pooling_conf.stride_w(), pooling_conf.pad_h(),
                pooling_conf.pad_w());
        break;
      }
      case PoolingOpConf::kStochastic: {
        TODO();
      }
      default: { UNEXPECTED_RUN(); }
    }
  }

  static void PoolingBackward(const KernelCtx& ctx, const Blob* out_diff_blob,
                              const Blob* mask_blob, Blob* in_diff_blob,
                              const PoolingOpConf& pooling_conf) {
    const int64_t count = in_diff_blob->shape().elem_cnt();

    switch (pooling_conf.pool()) {
      case PoolingOpConf::kMax: {
        MaxPoolBackward<T>
            <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
               ctx.device_ctx->cuda_stream()>>>(
                count, out_diff_blob->dptr<T>(), mask_blob->dptr<uint32_t>(),
                in_diff_blob->mut_dptr<T>(), in_diff_blob->shape().At(1),
                in_diff_blob->shape().At(2), in_diff_blob->shape().At(3),
                out_diff_blob->shape().At(2), out_diff_blob->shape().At(3),
                pooling_conf.kernel_h(), pooling_conf.kernel_w(),
                pooling_conf.stride_h(), pooling_conf.stride_w(),
                pooling_conf.pad_h(), pooling_conf.pad_w());
        break;
      }
      case PoolingOpConf::kAve: {
        AvePoolBackward<T>
            <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
               ctx.device_ctx->cuda_stream()>>>(
                count, out_diff_blob->dptr<T>(), in_diff_blob->mut_dptr<T>(),
                in_diff_blob->shape().At(1), in_diff_blob->shape().At(2),
                in_diff_blob->shape().At(3), out_diff_blob->shape().At(2),
                out_diff_blob->shape().At(3), pooling_conf.kernel_h(),
                pooling_conf.kernel_w(), pooling_conf.stride_h(),
                pooling_conf.stride_w(), pooling_conf.pad_h(),
                pooling_conf.pad_w());
        break;
      }
      case PoolingOpConf::kStochastic: {
        TODO();
      }
      default: { UNEXPECTED_RUN(); }
    }
  }
};

template<typename T>
CudnnPoolingKernel<DeviceType::kGPU, T>::CudnnPoolingKernel() {
  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
  CudaCheck(cudnnCreatePoolingDescriptor(&pooling_desc_));
}

template<typename T>
CudnnPoolingKernel<DeviceType::kGPU, T>::~CudnnPoolingKernel() {
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
  CudaCheck(cudnnDestroyPoolingDescriptor(pooling_desc_));
}

template<typename T>
void CudnnPoolingKernel<DeviceType::kGPU, T>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);

  const auto pooling_conf = op()->op_conf().pooling_conf();

  switch (pooling_conf.pool()) {
    case PoolingOpConf::kMax: {
      pooling_mode_ = CUDNN_POOLING_MAX;
      break;
    }
    case PoolingOpConf::kAve: {
      pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    }
    default: { UNEXPECTED_RUN(); }
  }

  CudaCheck(cudnnSetPooling2dDescriptor(
      pooling_desc_, pooling_mode_, CUDNN_PROPAGATE_NAN,
      pooling_conf.kernel_h(), pooling_conf.kernel_w(), pooling_conf.pad_h(),
      pooling_conf.pad_w(), pooling_conf.stride_h(), pooling_conf.stride_w()));
}

template<typename T>
void CudnnPoolingKernel<DeviceType::kGPU, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PoolingOpConf& pooling_conf = op()->op_conf().pooling_conf();

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CopyDataIdFromSoleIbToAllObIfNeed<DeviceType::kGPU>(ctx, BnInOp2Blob);

  CudaCheck(cudnnSetTensor4dDescriptor(
      in_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      in_blob->shape().At(0), in_blob->shape().At(1), in_blob->shape().At(2),
      in_blob->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      out_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      out_blob->shape().At(0), out_blob->shape().At(1), out_blob->shape().At(2),
      out_blob->shape().At(3)));

  CudaCheck(cudnnPoolingForward(ctx.device_ctx->cudnn_handle(), pooling_desc_,
                                cudnn::DataType<T>::one, in_desc_,
                                in_blob->dptr<T>(), cudnn::DataType<T>::zero,
                                out_desc_, out_blob->mut_dptr<T>()));
}

template<typename T>
void CudnnPoolingKernel<DeviceType::kGPU, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PoolingOpConf& pooling_conf = op()->op_conf().pooling_conf();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataField());

  CudaCheck(cudnnPoolingBackward(
      ctx.device_ctx->cudnn_handle(), pooling_desc_, cudnn::DataType<T>::one,
      out_desc_, out_blob->dptr<T>(), out_desc_, out_diff_blob->dptr<T>(),
      in_desc_, in_blob->dptr<T>(), cudnn::DataType<T>::zero, in_desc_,
      in_diff_blob->mut_dptr<T>()));
}

#ifdef USE_CUDNN
#define INSTANTIATE_POOLING_KERNEL(type_cpp, type_proto) \
  template class CudnnPoolingKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL, FLOATING_DATA_TYPE_SEQ)
#endif
#define INSTANTIATE_POOLING_KERNEL_UTIL(type_cpp, type_proto) \
  template class PoolingKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
