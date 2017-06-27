#include "oneflow/core/blas/kernel_util.h"
#include "oneflow/core/actor/cuda_device_context.h"
#include "oneflow/core/blas/cublas_template.h"

namespace oneflow {
template<typename floating_point_type> 
class KernelUtil<DeviceType::kGPU, floating_point_type> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);

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

  // static void BlasScal(const KernelCtx& ctx, const int n,
  //    const floating_point_type alpha, floating_point_type* x, int incx) {
  //  const floating_point_type alpha_0 = alpha;
  //  cublas_scal(ctx.device_ctx->cublas_handle(), n, alpha_0, x, incx);
  // }

};

template class KernelUtil<DeviceType::kGPU, float>;
template class KernelUtil<DeviceType::kGPU, double>;

}  // namespace oneflow
