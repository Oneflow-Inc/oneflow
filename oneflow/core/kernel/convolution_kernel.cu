#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {
template<typename T>
__global__ void Im2ColGpuKernel(const int n, const T* data_im, const int height,
                                const int width, const int kernel_h,
                                const int kernel_w, const int pad_h,
                                const int pad_w, const int stride_h,
                                const int stride_w, const int dilation_h,
                                const int dilation_w, const int height_col,
                                const int width_col, T* data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    T* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const T* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template<typename T>
__global__ void Col2ImGpuKernel(const int n, const T* data_col,
                                const int height, const int width,
                                const int channels, const int kernel_h,
                                const int kernel_w, const int pad_h,
                                const int pad_w, const int stride_h,
                                const int stride_w, const int dilation_h,
                                const int dilation_w, const int height_col,
                                const int width_col, T* data_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col)
                  * width_col
              + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

}  // namespace

template<typename T>
class ConvolutionKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernelUtil);
  ConvolutionKernelUtil() = delete;
  static void Im2Col(const KernelCtx& ctx, const T* data_im, const int channels,
                     const int height, const int width, const int kernel_h,
                     const int kernel_w, const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* data_col) {
    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    Im2ColGpuKernel<T>
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
            pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
            width_col, data_col);
  }

  static void Col2Im(const KernelCtx& ctx, const T* data_col,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* data_im) {
    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height * width;
    Col2ImGpuKernel<T>
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            height_col, width_col, data_im);
  }
};

#ifdef WITH_CUDNN
template<typename T>
void CudnnConvolutionKernel<T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto conv_conf = this->op_conf().convolution_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  const Blob* bias_blob = BnInOp2Blob("bias");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* cudnn_workspace = BnInOp2Blob("cudnn_workspace");

  this->cudnn_conv_desc_.InitFromBlobDescAndOpConf(
      in_blob->blob_desc_ptr(), out_blob->blob_desc_ptr(), conv_conf);

  this->cudnn_conv_desc_.Forward<T>(
      ctx.device_ctx->cudnn_handle(), in_blob, weight_blob, bias_blob, out_blob,
      cudnn_workspace,
      static_cast<cudnnConvolutionFwdAlgo_t>(
          this->kernel_conf().convolution_conf().cudnn_fwd_algo()),
      conv_conf.has_bias_term());
}

template<typename T>
void CudnnConvolutionKernel<T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto conv_conf = this->op_conf().convolution_conf();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");

  // compute bias_diff
  if (conv_conf.has_bias_term()) {
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
    Memset<DeviceType::kGPU>(ctx.device_ctx, bias_diff_blob->mut_dptr(), 0,
                             bias_diff_blob->ByteSizeOfDataContentField());
    this->cudnn_conv_desc_.BackwardBias<T>(ctx.device_ctx->cudnn_handle(),
                                           out_diff_blob, bias_diff_blob);
  }

  // compute weight_diff
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Blob* cudnn_workspace = BnInOp2Blob("cudnn_workspace");
  Memset<DeviceType::kGPU>(ctx.device_ctx, weight_diff_blob->mut_dptr(), 0,
                           weight_diff_blob->ByteSizeOfDataContentField());
  this->cudnn_conv_desc_.BackwardFilter<T>(
      ctx.device_ctx->cudnn_handle(), in_blob, out_diff_blob, weight_diff_blob,
      cudnn_workspace,
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->kernel_conf().convolution_conf().cudnn_bwd_filter_algo()));

  // compute in_diff
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());
  this->cudnn_conv_desc_.BackwardData<T>(
      ctx.device_ctx->cudnn_handle(), weight_blob, out_diff_blob, in_diff_blob,
      cudnn_workspace,
      static_cast<cudnnConvolutionBwdDataAlgo_t>(
          this->kernel_conf().convolution_conf().cudnn_bwd_data_algo()));
}

#define INSTANTIATE_CONVOLUTION_KERNEL(type_cpp, type_proto) \
  template class CudnnConvolutionKernel<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONVOLUTION_KERNEL, FLOATING_DATA_TYPE_SEQ)
#endif  // WITH_CUDNN

#define INSTANTIATE_CONVOLUTION_KERNEL_UTIL(type_cpp, type_proto) \
  template class ConvolutionKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONVOLUTION_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
