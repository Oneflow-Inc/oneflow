#include "oneflow/core/kernel/boxing_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<typename floating_point_type>
void BoxingKernel<DeviceType::kGPU, floating_point_type>::OFMemcpy(
    const KernelCtx& ctx, void* dst, const void* src, size_t sz) {
  CHECK_EQ(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, 
      ctx.device_ctx->cuda_stream()), cudaSuccess);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kGPU, floating_point_type>::OFBlobCpy(
    const KernelCtx& ctx, const Blob* a, Blob* b) {
  CHECK_EQ(cudaMemcpyAsync(static_cast<floating_point_type*>(b->mut_dptr()), \
        static_cast<const floating_point_type*>(a->dptr()), \
        sizeof(floating_point_type) * a->shape().elem_cnt(), \
        cudaMemcpyDeviceToDevice, ctx.device_ctx->cuda_stream()), \
      cudaSuccess);
}

template<>
void BoxingKernel<DeviceType::kGPU, float>::OFBlobAdd(
    const KernelCtx& ctx, const Blob* a, Blob* b) {
  static const float alpha = 1.0;
  CHECK_EQ(cublasSaxpy(
        ctx.device_ctx->cublas_handle(),
        a->shape().elem_cnt(), &alpha,
        static_cast<const float*>(a->dptr()), 1,
        static_cast<float*>(b->mut_dptr()), 1), \
      cudaSuccess);
}

template<>
void BoxingKernel<DeviceType::kGPU, double>::OFBlobAdd(
    const KernelCtx& ctx, const Blob* a, Blob* b) {
  static const double alpha = 1.0;
  CHECK_EQ(cublasDaxpy(
        ctx.device_ctx->cublas_handle(),
        a->shape().elem_cnt(), &alpha,
        static_cast<const double*>(a->dptr()), 1,
        static_cast<double*>(b->mut_dptr()), 1), 
      cudaSuccess);
}

template<>
void BoxingKernel<DeviceType::kGPU, float>::OFBlasAxpy(
    const KernelCtx& ctx, const int N, const float alpha, const float *X, \
    const int incX, float *Y, const int incY) {
  float tmp_alpha = alpha;
  CHECK_EQ(cublasSaxpy(
        ctx.device_ctx->cublas_handle(),
        N, &tmp_alpha, X, incX, Y, incY), 
      cudaSuccess);
}

template<>
void BoxingKernel<DeviceType::kGPU, double>::OFBlasAxpy(
    const KernelCtx& ctx, const int N, const double alpha, const double *X, \
    const int incX, double *Y, const int incY) {
  double tmp_alpha = alpha;
  CHECK_EQ(cublasDaxpy(
        ctx.device_ctx->cublas_handle(),
        N, &tmp_alpha, X, incX, Y, incY), 
      cudaSuccess);
}

template<> 
void BoxingKernel<DeviceType::kGPU, float>::OFBlasScal(
    const KernelCtx& ctx, const int n, const float alpha, float* x, int incx) {
  float tmp_alpha = alpha;
  CHECK_EQ(cublasSscal(
        ctx.device_ctx->cublas_handle(),
        n, &tmp_alpha, x, incx), 
      cudaSuccess);
}

template<> 
void BoxingKernel<DeviceType::kGPU, double>::OFBlasScal(
    const KernelCtx& ctx, const int n, const double alpha, double* x, \
    int incx) {
  double tmp_alpha = alpha;
  CHECK_EQ(cublasDscal(
        ctx.device_ctx->cublas_handle(),
        n, &tmp_alpha, x, incx), 
      cudaSuccess);
}

INSTANTIATE_GPU_KERNEL_CLASS(BoxingKernel);
REGISTER_GPU_KERNEL(OperatorConf::kBoxingConf, BoxingKernel);
}  // namespace oneflow
