#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/blas.h"

namespace oneflow {

void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float alpha, const float* a, const float* b,
const float beta, float* c) {
  FloatingOFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c, ldc)
}
void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const double alpha, const double* a, const double* b,
            const double beta, double* c) {
  DoubleOFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c, ldc);
}
void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
            const float16 beta, float16* c) {
  HalfOFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c, ldc);
}

static void FloatingOFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float alpha, const float* a, const float* b,
            const float beta, float* c) {
  Gemm<float>(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

static void DoubleOFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const double alpha, const double* a, const double* b,
            const float16 beta, float16* c) {
  Gemm<double>(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

static void HalfOFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
            const float16 beta, float16* c) {
  UNIMPLEMENTED()
}

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c, const int ldc) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cblas_gemm<T>(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
} // namespace oneflow

