#include "oneflow/core/blas/kernel_util.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {

template<>
void KernelUtil<DeviceType::kGPU, float>::Memcpy(
    const KernelCtx& ctx, void* dst, const void* src, size_t sz) {
  CHECK_EQ(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, 
      ctx.device_ctx->cuda_stream()), cudaSuccess);
}

template<>
void KernelUtil<DeviceType::kGPU, double>::Memcpy(
    const KernelCtx& ctx, void* dst, const void* src, size_t sz) {
  CHECK_EQ(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, 
      ctx.device_ctx->cuda_stream()), cudaSuccess);
}

template<>
void KernelUtil<DeviceType::kGPU, float>::Memset(
    const KernelCtx& ctx, void* dst, const char value, size_t sz) {
  CHECK_EQ(cudaMemsetAsync(dst, value, sz, ctx.device_ctx->cuda_stream()), 
      cudaSuccess);
}

template<>
void KernelUtil<DeviceType::kGPU, double>::Memset(
    const KernelCtx& ctx, void* dst, const char value, size_t sz) {
  CHECK_EQ(cudaMemsetAsync(dst, value, sz, ctx.device_ctx->cuda_stream()), 
      cudaSuccess);
}

template<>
void KernelUtil<DeviceType::kGPU, float>::BlasAxpy(
    const KernelCtx& ctx, const int N, const float alpha, const float *X, 
    const int incX, float *Y, const int incY) {
  float tmp_alpha = alpha;
  CHECK_EQ(cublasSaxpy(
        ctx.device_ctx->cublas_handle(),
        N, &tmp_alpha, X, incX, Y, incY), 
      cudaSuccess);
}

template<>
void KernelUtil<DeviceType::kGPU, double>::BlasAxpy(
    const KernelCtx& ctx, const int N, const double alpha, const double *X, 
    const int incX, double *Y, const int incY) {
  double tmp_alpha = alpha;
  CHECK_EQ(cublasDaxpy(
        ctx.device_ctx->cublas_handle(),
        N, &tmp_alpha, X, incX, Y, incY), 
      cudaSuccess);
}

template<> 
void KernelUtil<DeviceType::kGPU, float>::BlasScal(
    const KernelCtx& ctx, const int n, const float alpha, float* x, int incx) {
  float tmp_alpha = alpha;
  CHECK_EQ(cublasSscal(
        ctx.device_ctx->cublas_handle(),
        n, &tmp_alpha, x, incx), 
      cudaSuccess);
}

template<> 
void KernelUtil<DeviceType::kGPU, double>::BlasScal(
    const KernelCtx& ctx, const int n, const double alpha, double* x,
    int incx) {
  double tmp_alpha = alpha;
  CHECK_EQ(cublasDscal(
        ctx.device_ctx->cublas_handle(),
        n, &tmp_alpha, x, incx), 
      cudaSuccess);
}

}  // namespace oneflow
