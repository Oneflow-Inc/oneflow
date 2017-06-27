#include "oneflow/core/blas/kernel_util.h"

namespace oneflow {

template<typename floating_point_type> 
class KernelUtil<DeviceType::kCPU, floating_point_type> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  static void Memcpy(const KernelCtx& ctx, 
     void* dst, const void* src, size_t sz) {
    ctx.device_ctx->cpu_stream()->Send([dst, src, sz](){
      memcpy(dst, src, sz);
    });
  }

  static void Memset(const KernelCtx& ctx, void* dst, const char value,
      size_t sz) {
    ctx.device_ctx->cpu_stream()->Send([dst, value, sz](){
      memset(dst, value, sz);
    });
  }

  static void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY) {
    ctx.device_ctx->cpu_stream()->Send([N, alpha, X, incX, Y, incY]() {
      cblas_axpy(N, alpha, X, incX, Y, incY);
    });
  }

  static void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, int incx) {
    ctx.device_ctx->cpu_stream()->Send([n, alpha, x, incx]() {
      cblas_scal(n, alpha, x, incx);
    });
  }

  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans, 
      int m, int n, const floating_point_type alpha, 
      const floating_point_type* A, int lda, const floating_point_type* x, 
      int incx, const floating_point_type beta, 
      floating_point_type* y, int incy) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      // Set col major to keep it as the same with cublas
      cblas_gemv(CBLAS_ORDER::CblasColMajor, trans, m, n, alpha, A, lda, x,
        incx, beta, y, incy);
    });
  }

  static void BlasGemm(const KernelCtx& ctx,
      const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
      const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
      const floating_point_type alpha, const floating_point_type* A,
      const int lda, const floating_point_type* B, const int ldb,
      const floating_point_type beta, floating_point_type* C, const int ldc) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      cblas_gemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
          C, ldc);
    });
  }

  static void BlasDot(const KernelCtx& ctx, 
      const int N, const floating_point_type* X, const int incX,
      const floating_point_type* Y, const int incY, 
      floating_point_type* result) {
    ctx.device_ctx->cpu_stream()->Send([=]() {
      *result = cblas_dot(N, X, incX, Y, incY);
    });
  }

  static void BlasSwap(const KernelCtx& ctx,
      const int N,
      floating_point_type* X, const int incX,
      floating_point_type* Y, const int incY) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      cblas_swap(N, X, incX, Y, incY);     
    });
  }

  static void BlasCopy(const KernelCtx& ctx,
      const int N,
      const floating_point_type* X, const int incX,
      floating_point_type* Y, const int incY) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      cblas_copy(N, X, incX, Y, incY);
    });
  }
};

template class KernelUtil<DeviceType::kCPU, float>;
template class KernelUtil<DeviceType::kCPU, double>;

}  //  namespace oneflow
