#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {
template<typename FloatingPointType>
__global__ void Im2ColGpuKernel(const int n, const FloatingPointType* data_im,
                                const int height, const int width,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w,
                                const int stride_h, const int stride_w,
                                const int dilation_h, const int dilation_w,
                                const int height_col, const int width_col,
                                FloatingPointType* data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    FloatingPointType* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const FloatingPointType* data_im_ptr = data_im;
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

template<typename FloatingPointType>
__global__ void Col2ImGpuKernel(
    const int n, const FloatingPointType* data_col, const int height,
    const int width, const int channels, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int height_col,
    const int width_col, FloatingPointType* data_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    FloatingPointType val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
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

template<typename FloatingPointType>
class ConvolutionKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernelUtil);
  ConvolutionKernelUtil() = delete;
  static void Im2Col(const KernelCtx& ctx, const FloatingPointType* data_im,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     FloatingPointType* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    Im2ColGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
            pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
            width_col, data_col);
  }

  static void Col2Im(const KernelCtx& ctx, const FloatingPointType* data_col,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     FloatingPointType* data_im) {
    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height * width;
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    Col2ImGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            height_col, width_col, data_im);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(ConvolutionKernelUtil);
}  // namespace oneflow
