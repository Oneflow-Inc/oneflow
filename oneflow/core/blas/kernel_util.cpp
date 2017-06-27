#include "oneflow/core/blas/kernel_util.h"
#include "oneflow/core/blas/cblas_template.h"

namespace oneflow {

template<>
void KernelUtil<DeviceType::kCPU, float>::BlasAxpy(
    const KernelCtx& ctx, 
    const int N, const float alpha, 
    const float *X, const int incX, 
    float *Y, const int incY) {
  cblas_axpy(N, alpha, X, incX, Y, incY);
}

template<>
void KernelUtil<DeviceType::kCPU, double>::BlasAxpy(
    const KernelCtx& ctx, 
    const int N, const double alpha, 
    const double *X, const int incX, 
    double *Y, const int incY) {
  cblas_axpy(N, alpha, X, incX, Y, incY);
}

template<> 
void KernelUtil<DeviceType::kCPU, double>::BlasScal(
    const KernelCtx& ctx,
    const int n, const double alpha,
    double* x, int incx) {
  cblas_scal(n, alpha, x, incx);
}

template<> 
void KernelUtil<DeviceType::kCPU, float>::BlasScal(
    const KernelCtx& ctx,
    const int n, const float alpha,
    float* x, int incx) {
  cblas_scal(n, alpha, x, incx);
}

template<>
void KernelUtil<DeviceType::kCPU, float>::Memcpy(
    const KernelCtx& ctx, void* dst,
    const void* src, size_t sz) {
  memcpy(dst, src, sz);
}

template<>
void KernelUtil<DeviceType::kCPU, double>::Memcpy(
    const KernelCtx& ctx, void* dst,
    const void* src, size_t sz) {
  memcpy(dst, src, sz);
}

template<>
void KernelUtil<DeviceType::kCPU, float>::Memset(
    const KernelCtx& ctx, void* dst, const char value, size_t sz) {
  memset(dst, value, sz);
}

template<>
void KernelUtil<DeviceType::kCPU, double>::Memset(
    const KernelCtx& ctx, void* dst, const char value, size_t sz) {
  memset(dst, value, sz);
}

}  //  namespace oneflow
