#include "oneflow/core/kernel/util/host_blas_interface.h"

namespace oneflow {

namespace {

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
                 enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const T alpha,
                 const T* a, const T* b, const T beta, T* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cblas_gemm<T>(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<typename T>
static void BlobGemmImpl(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                         T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().Count(1);
  const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

  BlasIf<kCPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
                       c->mut_dptr<T>());
}

template<typename T>
static void AxpyImpl(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                     const int incy) {
  FOR_RANGE(int, i, 0, n) {
    *y += alpha * *x;
    x += incx;
    y += incy;
  }
}

template<typename T>
void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                     int batch_size, int m, int n, int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c, T** buf) {
  const int a_stride = m * k;
  const int b_stride = k * n;
  const int c_stride = m * n;
  FOR_RANGE(int32_t, i, 0, batch_size) {
    BlasIf<DeviceType::kCPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a + i * a_stride,
                                     b + i * b_stride, beta, c + i * c_stride);
  }
}

}  // namespace

void BlasIf<DeviceType::kCPU>::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                        enum CBLAS_TRANSPOSE trans_b, float alpha, float beta,
                                        const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<float>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

void BlasIf<DeviceType::kCPU>::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                        enum CBLAS_TRANSPOSE trans_b, double alpha, double beta,
                                        const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<double>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

void BlasIf<DeviceType::kCPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const float alpha, const float* a,
                                      const float* b, const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

void BlasIf<DeviceType::kCPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const double alpha, const double* a,
                                      const double* b, const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

void BlasIf<DeviceType::kCPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const float alpha, const float* a, const float* b,
                                             const float beta, float* c, float** buf) {
  BatchedGemmImpl<float>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                         beta, c, buf);
}

void BlasIf<DeviceType::kCPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const double alpha, const double* a, const double* b,
                                             const double beta, double* c, double** buf) {
  BatchedGemmImpl<double>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                          beta, c, buf);
}

void BlasIf<DeviceType::kCPU>::Axpy(DeviceCtx* ctx, const int n, const float alpha, const float* x,
                                    const int incx, float* y, const int incy) {
  AxpyImpl<float>(ctx, n, alpha, x, incx, y, incy);
}

void BlasIf<DeviceType::kCPU>::Axpy(DeviceCtx* ctx, const int n, const double alpha,
                                    const double* x, const int incx, double* y, const int incy) {
  AxpyImpl<double>(ctx, n, alpha, x, incx, y, incy);
}

}  // namespace oneflow
