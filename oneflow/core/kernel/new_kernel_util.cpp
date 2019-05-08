#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/kernel/kernel.h"

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

  NewKernelUtil<kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
}

template<typename T>
static void ReluImpl(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  T zero = ZeroVal<T>::value;
  for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
}

template<typename T>
static void ReluBackwardImpl(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
  T zero = ZeroVal<T>::value;
  for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
}

} // namespace

#define CPU_KU_METHOD void NewKernelUtil<DeviceType::kCPU>::

CPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                      float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<float>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

CPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       double alpha, double beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<double>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float alpha, const float* a, const float* b,
            const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const double alpha, const double* a, const double* b,
            const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
            const float16 beta, float16* c) {
   UNIMPLEMENTED();
}

CPU_KU_METHOD Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y) {
  ReluImpl<float>(ctx, n, x, y);
}

CPU_KU_METHOD Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y) {
  ReluImpl<double>(ctx, n, x, y);
}

CPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y, const float* dy,
                           float* dx) {
  ReluBackwardImpl<float>(ctx, n, x, y, dy, dx);
}

CPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y, const double* dy,
                           double* dx) {
  ReluBackwardImpl<double>(ctx, n, x, y, dy, dx);
}

} // namespace oneflow

