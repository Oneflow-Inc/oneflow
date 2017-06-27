#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
template<typename floating_point_type> 
class KernelUtil<DeviceType::kGPU, floating_point_type> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  static void Memcpy(const KernelCtx& ctx, 
     void* dst, const void* src, size_t sz) {
    CHECK_EQ(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, 
        ctx.device_ctx->cuda_stream()), cudaSuccess);
  }

  static void Memset(const KernelCtx& ctx, void* dst, const char value,
      size_t sz) {
    CHECK_EQ(cudaMemsetAsync(dst, value, sz, ctx.device_ctx->cuda_stream()), 
        cudaSuccess);
  }

  static void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY) {
    const floating_point_type alpha_0 = alpha;
    cublas_axpy(ctx.device_ctx->cublas_handle(), N, &alpha_0, X, incX, Y, incY);
  }

  static void BlasScal(const KernelCtx& ctx, const int n,
     const floating_point_type alpha, floating_point_type* x, const int incx) {
   const floating_point_type alpha_0 = alpha;
   cublas_scal(ctx.device_ctx->cublas_handle(), n, &alpha_0, x, incx);
  }

  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans, 
      int m, int n, const floating_point_type alpha, 
      const floating_point_type* A, int lda, const floating_point_type* x, 
      const int incx, const floating_point_type beta, floating_point_type* y, 
      const int incy) {
    cublasOperation_t cublas_trans = CblasTrans2CublasTrans(trans);
    const floating_point_type alpha_0 = alpha;
    const floating_point_type beta_0 = beta;
    cublas_gemv(ctx.device_ctx->cublas_handle(), cublas_trans, m, n, 
        &alpha_0, A, lda, x, incx, &beta_0, y, incy);
  }

  static void BlasGemm(const KernelCtx& ctx,
      const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
      const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
      const floating_point_type alpha, const floating_point_type* A,
      const int lda, const floating_point_type* B, const int ldb,
      const floating_point_type beta, floating_point_type* C, const int ldc) {
    cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(TransA);
    cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(TransB);
    const floating_point_type alpha_0 = alpha;
    const floating_point_type beta_0 = beta;
    cublas_gemm(ctx.device_ctx->cublas_handle(), cublas_trans_a, cublas_trans_b,
        M, N, K, &alpha_0, A, lda, B, ldb, &beta_0, C, ldc);
  }

  static void BlasDot(const KernelCtx& ctx,
      const int N, const floating_point_type* X, const int incX,
      const floating_point_type* Y, const int incY, 
      floating_point_type* result) {
    cublas_dot(ctx.device_ctx->cublas_handle(), N, X, incX, Y, incY, result);
  }

  static void BlasSwap(const KernelCtx& ctx,
      const int N,
      floating_point_type* X, const int incX,
      floating_point_type* Y, const int incY) {
    cublas_swap(ctx.device_ctx->cublas_handle(), N, X, incX, Y, incY);
  }

  static void BlasCopy(const KernelCtx& ctx,
      const int N,
      const floating_point_type* X, const int incX,
      floating_point_type* Y, const int incY) {
    cublas_copy(ctx.device_ctx->cublas_handle(), N, X, incX, Y, incY);
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

template class KernelUtil<DeviceType::kGPU, float>;
template class KernelUtil<DeviceType::kGPU, double>;

}  // namespace oneflow
