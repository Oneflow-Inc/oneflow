#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename floating_point_type> 
class KernelUtil<DeviceType::kCPU, floating_point_type> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  static void Memcpy(const KernelCtx& ctx, 
     void* dst, const void* src, size_t sz, 
     cudaMemcpyKind kind=cudaMemcpyKind::cudaMemcpyHostToHost) {
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

  static void BlasAxpy(const KernelCtx& ctx, const int n,
      const floating_point_type alpha,
      const floating_point_type* x, const int incx,
      floating_point_type* y, const int incy) {
    ctx.device_ctx->cpu_stream()->Send([n, alpha, x, incx, y, incy]() {
      cblas_axpy(n, alpha, x, incx, y, incy);
    });
  }

  static void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, const int incx) {
    ctx.device_ctx->cpu_stream()->Send([n, alpha, x, incx]() {
      cblas_scal(n, alpha, x, incx);
    });
  }

  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans, 
      int m, int n, const floating_point_type alpha, 
      const floating_point_type* a, int lda, const floating_point_type* x, 
      const int incx, const floating_point_type beta, 
      floating_point_type* y, const int incy) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      // Set col major to keep it as the same with cublas
      cblas_gemv(CBLAS_ORDER::CblasColMajor, trans, m, n, alpha, a, lda, x,
        incx, beta, y, incy);
    });
  }

  static void BlasGemm(const KernelCtx& ctx,
      const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
      const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
      const floating_point_type alpha, const floating_point_type* a,
      const int lda, const floating_point_type* b, const int ldb,
      const floating_point_type beta, floating_point_type* c, const int ldc) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      cblas_gemm(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta,
          c, ldc);
    });
  }

  static void BlasDot(const KernelCtx& ctx, 
      const int n, const floating_point_type* x, const int incx,
      const floating_point_type* y, const int incy, 
      floating_point_type* result) {
    ctx.device_ctx->cpu_stream()->Send([=]() {
      *result = cblas_dot(n, x, incx, y, incy);
    });
  }

  static void BlasSwap(const KernelCtx& ctx,
      const int n,
      floating_point_type* x, const int incx,
      floating_point_type* y, const int incy) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      cblas_swap(n, x, incx, y, incy);     
    });
  }

  static void BlasCopy(const KernelCtx& ctx,
      const int n,
      const floating_point_type* x, const int incx,
      floating_point_type* y, const int incy) {
    ctx.device_ctx->cpu_stream()->Send([=](){
      cblas_copy(n, x, incx, y, incy);
    });
  }
};

template class KernelUtil<DeviceType::kCPU, float>;
template class KernelUtil<DeviceType::kCPU, double>;

}  //  namespace oneflow
