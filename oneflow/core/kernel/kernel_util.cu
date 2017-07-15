#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
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

template<typename FloatingPointType>
__global__ void ExpGpu(const int64_t n, const FloatingPointType* x,
                       FloatingPointType* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::exp(x[i]); }
}

template<typename FloatingPointType>
__global__ void DivGpu(const int64_t n, FloatingPointType* x,
                       const FloatingPointType alpha) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = x[i] / alpha; }
}

template<typename FloatingPointType>
__global__ void MulGpu(const int64_t n, const FloatingPointType* x,
                       const FloatingPointType* y, FloatingPointType* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[i]; }
}

}  // namespace

template<typename FloatingPointType>
class KernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  static void Memcpy(const KernelCtx& ctx, void* dst, const void* src,
                     size_t sz, cudaMemcpyKind kind) {
    CudaCheck(
        cudaMemcpyAsync(dst, src, sz, kind, ctx.device_ctx->cuda_stream()));
  }

  static void Memset(const KernelCtx& ctx, void* dst, const char value,
                     size_t sz) {
    CudaCheck(cudaMemsetAsync(dst, value, sz, ctx.device_ctx->cuda_stream()));
  }

  static void BlasAxpy(const KernelCtx& ctx, const int n,
                       const FloatingPointType alpha,
                       const FloatingPointType* x, const int incx,
                       FloatingPointType* y, const int incy) {
    cublas_axpy(ctx.device_ctx->cublas_handle(), n, &alpha, x, incx, y, incy);
  }

  static void BlasScal(const KernelCtx& ctx, const int n,
                       const FloatingPointType alpha, FloatingPointType* x,
                       const int incx) {
    cublas_scal(ctx.device_ctx->cublas_handle(), n, &alpha, x, incx);
  }

  static void Exp(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* y) {
    ExpGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, x, y);
  }

  static void Div(const KernelCtx& ctx, const int64_t n, FloatingPointType* x,
                  const FloatingPointType alpha) {
    DivGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, x, alpha);
  }

  static void Mul(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, const FloatingPointType* y,
                  FloatingPointType* z) {
    MulGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, x, y, z);
  }

  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans,
                       int m, int n, const FloatingPointType alpha,
                       const FloatingPointType* a, int lda,
                       const FloatingPointType* x, const int incx,
                       const FloatingPointType beta, FloatingPointType* y,
                       const int incy) {
    cublasOperation_t cublas_trans = CblasTrans2CublasTrans(trans);
    cublas_gemv(ctx.device_ctx->cublas_handle(), cublas_trans, n, m, &alpha, a,
                lda, x, incx, &beta, y, incy);
  }

  static void BlasGemm(const KernelCtx& ctx, const enum CBLAS_ORDER order,
                       const enum CBLAS_TRANSPOSE trans_a,
                       const enum CBLAS_TRANSPOSE trans_b, const int m,
                       const int n, const int k, const FloatingPointType alpha,
                       const FloatingPointType* a, const int lda,
                       const FloatingPointType* b, const int ldb,
                       const FloatingPointType beta, FloatingPointType* c,
                       const int ldc) {
    cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
    cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
    cublas_gemm(ctx.device_ctx->cublas_handle(), cublas_trans_b, cublas_trans_a,
                n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc);
  }

  static void BlasDot(const KernelCtx& ctx, const int n,
                      const FloatingPointType* x, const int incx,
                      const FloatingPointType* y, const int incy,
                      FloatingPointType* result) {
    cublas_dot(ctx.device_ctx->cublas_handle(), n, x, incx, y, incy, result);
  }

  static void BlasSwap(const KernelCtx& ctx, const int n, FloatingPointType* x,
                       const int incx, FloatingPointType* y, const int incy) {
    cublas_swap(ctx.device_ctx->cublas_handle(), n, x, incx, y, incy);
  }

  static void BlasCopy(const KernelCtx& ctx, const int n,
                       const FloatingPointType* x, const int incx,
                       FloatingPointType* y, const int incy) {
    cublas_copy(ctx.device_ctx->cublas_handle(), n, x, incx, y, incy);
  }

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
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock>>>(
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
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock>>>(
            num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            height_col, width_col, data_im);
  }

  static void Fill(const KernelCtx& ctx, const FillConf& fill_conf,
                   Blob* blob) {
    void* host_raw_dptr;
    size_t byte_size = blob->shape().elem_cnt() * sizeof(FloatingPointType);
    CudaCheck(cudaMallocHost(&host_raw_dptr, byte_size));

    std::unique_ptr<void, std::function<void(void*)>> host_unique_ptr(
        host_raw_dptr, [&](void* dptr) { CudaCheck(cudaFree(dptr)); });
    std::unique_ptr<Shape> host_blob_shape(new Shape(blob->shape()));

    std::unique_ptr<Blob> host_blob(
        new Blob(host_unique_ptr.get(), host_blob_shape.get()));
    KernelUtil<DeviceType::kCPU, FloatingPointType>::Fill(ctx, fill_conf,
                                                          host_blob.get());

    KernelUtil<DeviceType::kGPU, FloatingPointType>::Memcpy(
        ctx, blob->mut_dptr(), host_blob->dptr(), byte_size,
        cudaMemcpyHostToDevice);
  }

 private:
  static cublasOperation_t CblasTrans2CublasTrans(CBLAS_TRANSPOSE trans) {
    cublasOperation_t cublas_trans;
    if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
      cublas_trans = cublasOperation_t::CUBLAS_OP_N;
    } else if (trans == CBLAS_TRANSPOSE::CblasTrans) {
      cublas_trans = cublasOperation_t::CUBLAS_OP_T;
    } else if (trans == CBLAS_TRANSPOSE::CblasConjTrans) {
      cublas_trans = cublasOperation_t::CUBLAS_OP_C;
    } else {
      // do nothing
    }
    return cublas_trans;
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(KernelUtil);
}  // namespace oneflow
