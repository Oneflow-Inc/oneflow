#include "oneflow/core/kernel/boxing_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<typename floating_point_type>
void BoxingKernel<DeviceType::kGPU, floating_point_type>::OFMemcpy(
    const KernelCtx& ctx, void* dst, 
    const void* src, size_t sz) {
  CHECK_EQ(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, 
      ctx.device_ctx->cuda_stream()), cudaSuccess);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kGPU, floating_point_type>::OFAddBlob(
    const KernelCtx& ctx, Blob* a, Blob* b) {
  CHECK_EQ(cublas_axpy<floating_point_type>(
        ctx.device()->cublas_handle(),
        a->shape().elem_cnt(), 1.0,
        static_cast<const floating_point_type>(a->dptr()), 1,
        static_cast<floating_point_type>(b->mut_dptr()), 1), 
      cudaSuccess);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kGPU, floating_point_type>::of_cblas_axpy(
    const KernelCtx& ctx, 
    const int N, const floating_point_type alpha, 
    const floating_point_type *X, const int incX, 
    floating_point_type *Y, const int incY) {
  CHECK_EQ(cublas_axpy<floating_point_type>(
        ctx.device()->cublas_handle(),
        N, alpha, X, incX, Y, incY), 
      cudaSuccess);
}

template<typename floating_point_type> 
void BoxingKernel<DeviceType::kGPU, floating_point_type>::of_cblas_scal(
    const KernelCtx& ctx, 
    const int n, const floating_point_type alpha,
    floating_point_type* x, int incx) {
  CHECK_EQ(cublas_scal<floating_point_type>(
        ctx.device()->cublas_handle(),
        n, alpha, x, incx), 
      cudaSuccess);

}

//} // namespace

INSTANTIATE_GPU_KERNEL_CLASS(BoxingKernel);
REGISTER_GPU_KERNEL(OperatorConf::kBoxingConf, BoxingKernel);
} // namespace oneflow
