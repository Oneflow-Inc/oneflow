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

  static void BlasAxpy(const KernelCtx& ctx, const int n,
      const floating_point_type alpha,
      const floating_point_type* x, const int incx,
      floating_point_type* y, const int incy) {
    cublas_axpy(ctx.device_ctx->cublas_handle(), n, &alpha, x, incx, y, incy);
  }

  static void BlasScal(const KernelCtx& ctx, const int n,
     const floating_point_type alpha, floating_point_type* x, const int incx) {
   cublas_scal(ctx.device_ctx->cublas_handle(), n, &alpha, x, incx);
  }

  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans, 
      int m, int n, const floating_point_type alpha, 
      const floating_point_type* a, int lda, const floating_point_type* x, 
      const int incx, const floating_point_type beta, floating_point_type* y, 
      const int incy) {
    cublasOperation_t cublas_trans = CblasTrans2CublasTrans(trans);
    cublas_gemv(ctx.device_ctx->cublas_handle(), cublas_trans, n, m, 
        &alpha, a, lda, x, incx, &beta, y, incy);
  }

  static void BlasGemm(const KernelCtx& ctx,
      const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
      const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
      const floating_point_type alpha, const floating_point_type* a,
      const int lda, const floating_point_type* b, const int ldb,
      const floating_point_type beta, floating_point_type* c, const int ldc) {
    cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
    cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
    cublas_gemm(ctx.device_ctx->cublas_handle(), cublas_trans_b, cublas_trans_a,
        n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc);
  }

  static void BlasDot(const KernelCtx& ctx,
      const int n, const floating_point_type* x, const int incx,
      const floating_point_type* y, const int incy, 
      floating_point_type* result) {
    cublas_dot(ctx.device_ctx->cublas_handle(), n, x, incx, y, incy, result);
  }

  static void BlasSwap(const KernelCtx& ctx,
      const int n,
      floating_point_type* x, const int incx,
      floating_point_type* y, const int incy) {
    cublas_swap(ctx.device_ctx->cublas_handle(), n, x, incx, y, incy);
  }

  static void BlasCopy(const KernelCtx& ctx,
      const int n,
      const floating_point_type* x, const int incx,
      floating_point_type* y, const int incy) {
    cublas_copy(ctx.device_ctx->cublas_handle(), n, x, incx, y, incy);
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
