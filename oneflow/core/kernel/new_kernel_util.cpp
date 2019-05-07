#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/blas.h"

namespace oneflow {

namespace {

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const T alpha, const T* a, const T* b,
            const T beta, T* c) {
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

  NewKernelUtil::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
}

template<typename T>
static void ReluImpl(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
}

template<typename T>
static void ReluBackwardImpl(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    T zero = ZeroVal<T>::value;
    for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
}

} // namespace

void NewKernelUtil::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                      float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

void NewKernelUtil::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       double alpha, double beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

void NewKernelUtil::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                      float16 alpha, float16 beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float alpha, const float* a, const float* b,
            const float beta, float* c) {
  Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const double alpha, const double* a, const double* b,
            const double beta, double* c) {
  Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
            const float16 beta, float16* c) {
   UNIMPLEMENTED();
}

void NewKernelUtil::Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y) {
  ReluImpl(ctx, n, x, y);
}

void NewKernelUtil::Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y) {
  ReluImpl(ctx, n, x, y);
}

void NewKernelUtil::Relu(DeviceCtx* ctx, const int64_t n, const float16* x, float16* y) {
  ReluImpl(ctx, n, x, y);
}

void NewKernelUtil::ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y, const float* dy,
                           float* dx) {
  ReluBackwardImpl(ctx, n, x, y, dy, dx);
}

void NewKernelUtil::ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y, const double* dy,
                           double* dx) {
  ReluBackwardImpl(ctx, n, x, y, dy, dx);
}

void NewKernelUtil::ReluBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y, const float16* dy,
                           float16* dx) {
   ReluBackwardImpl(ctx, n, x, y, dy, dx);
}

} // namespace oneflow

