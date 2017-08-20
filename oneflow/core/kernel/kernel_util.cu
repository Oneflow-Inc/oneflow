#include "cub/cub.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void ExpGpu(const int64_t n, const FloatingPointType* x,
                       FloatingPointType* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::exp(x[i]); }
}

template<typename FloatingPointType>
__global__ void DivGpu(const int64_t n, FloatingPointType* x,
                       const FloatingPointType* alpha_ptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = x[i] / (*alpha_ptr); }
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

  static void Max(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* max_ptr,
                  FloatingPointType* temp_storage, size_t temp_storage_bytes) {
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, x, max_ptr, n,
                           ctx.device_ctx->cuda_stream());
  }

  static void Exp(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* y) {
    ExpGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, x, y);
  }

  static void Sum(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* sum_ptr,
                  FloatingPointType* temp_storage, size_t temp_storage_bytes) {
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, x, sum_ptr, n,
                           ctx.device_ctx->cuda_stream());
  }

  static void Div(const KernelCtx& ctx, const int64_t n, FloatingPointType* x,
                  const FloatingPointType* alpha_ptr) {
    DivGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, x, alpha_ptr);
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

  static void Fill(const KernelCtx& ctx, const FillConf& fill_conf,
                   uint32_t random_seed, Blob* blob) {
    TODO();
  }

  static void FillWithSnapshot(const KernelCtx& ctx, int32_t part_id,
                               int32_t part_num, const Snapshot* snapshot,
                               Blob* blob, const std::string& lbn,
                               int32_t dim_num, int64_t num_in_each_dim) {
    TODO();
    /*
    int64_t blob_size = blob->shape().elem_cnt() * sizeof(FloatingPointType);
    std::unique_ptr<PersistentInStream> in_stream =
        snapshot->GetInStream(lbn, part_id, part_num, dim_num,
                              num_in_each_dim * sizeof(FloatingPointType));
    // read model from disk to host_blob synchronously
    void* host_raw_dptr;
    CudaCheck(cudaMallocHost(&host_raw_dptr, blob_size));
    std::unique_ptr<void, std::function<void(void*)>> host_unique_ptr(
        host_raw_dptr, [&](void* dptr) { CudaCheck(cudaFreeHost(dptr)); });
    std::unique_ptr<Shape> host_blob_shape(new Shape(blob->shape()));
    std::unique_ptr<Blob> host_blob(
        new Blob(host_unique_ptr.get(), host_blob_shape.get()));
    in_stream->Read(host_blob->mut_dptr<char>(), blob_size);
    // copy to device blob
    KernelUtil<DeviceType::kGPU, FloatingPointType>::Memcpy(
        ctx, blob->mut_dptr(), host_blob->dptr(), blob_size,
        cudaMemcpyHostToDevice);
        */
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
