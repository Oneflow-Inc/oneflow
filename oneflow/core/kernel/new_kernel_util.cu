#include "oneflow/core/kernel/new_kernel_util.h"
#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {
  
cublasOperation_t CblasTrans2CublasTrans(CBLAS_TRANSPOSE trans) {
  cublasOperation_t cublas_trans;
  if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_N;
  } else if (trans == CBLAS_TRANSPOSE::CblasTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_T;
  } else if (trans == CBLAS_TRANSPOSE::CblasConjTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_C;
  } else {
    // do nothing
  }
  return cublas_trans;
}

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const T alpha, const T* a, const T* b,
  const T beta, T* c) {  
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);

  cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b,
  ldb, a, lda, &beta, c, ldc);
}

static void HGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
  const float16 beta, float16* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  CudaCheck(cublasHgemm(ctx->cublas_tensor_op_math_handle(), cublas_trans_b, cublas_trans_a, n, m,
        k, reinterpret_cast<const half*>(&alpha),
        reinterpret_cast<const half*>(b), ldb, reinterpret_cast<const half*>(a),
        lda, reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c),
        ldc));
}

} // namespace

void NewKernelUtil::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  NewKernelUtil::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
void NewKernelUtil::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  NewKernelUtil::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
void NewKernelUtil::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  NewKernelUtil::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const float alpha, const float* a, const float* b,
const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const double alpha, const double* a, const double* b,
    const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
    const float16 beta, float16* c) {
  HGemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

} // namespace oneflow
