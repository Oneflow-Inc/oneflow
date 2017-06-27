#include "oneflow/core/blas/kernel_util.h"
#include "oneflow/core/blas/cblas_template.h"

namespace oneflow {

template<typename floating_point_type> 
class KernelUtil<DeviceType::kCPU, floating_point_type> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);

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
    ctx.device_ctx->cpu_stream()->Send([n,alpha, x, incx]() {
      cblas_scal(n, alpha, x, incx);
    });
  }
};

template class KernelUtil<DeviceType::kCPU, float>;
template class KernelUtil<DeviceType::kCPU, double>;

}  //  namespace oneflow
