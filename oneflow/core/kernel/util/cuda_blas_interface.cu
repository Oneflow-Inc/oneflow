/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/kernel/util/cuda_blas_interface.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

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

std::tuple<int, int, int, cublasOperation_t, cublasOperation_t> PrepareToCallCublasGemm(
    enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
    const int k) {
  int lda = (trans_a == CblasNoTrans) ? k : m;
  int ldb = (trans_b == CblasNoTrans) ? n : k;
  int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  return std::make_tuple(lda, ldb, ldc, cublas_trans_a, cublas_trans_b);
}

template<typename T>
void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const T* alpha,
          const T* a, const T* b, const T* beta, T* c) {
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallCublasGemm(trans_a, trans_b, m, n, k);

  cublasHandle_t handle;
  if (std::is_same<T, half>::value) {
    handle = ctx->cublas_tensor_op_math_handle();
  } else {
    handle = ctx->cublas_pmh_handle();
  }
  cublas_gemm<T>(handle, cublas_trans_b, cublas_trans_a, n, m, k, alpha, b, ldb, a, lda, beta, c,
                 ldc);
}

template<>
void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const half* alpha,
          const half* a, const half* b, const half* beta, half* c) {
  const float alpha_f = __half2float(*alpha);
  const float beta_f = __half2float(*beta);
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallCublasGemm(trans_a, trans_b, m, n, k);
  if (GetCudaSmVersion() >= 500) {
    OF_CUBLAS_CHECK(cublasGemmEx(ctx->cublas_tensor_op_math_handle(), cublas_trans_b,
                                 cublas_trans_a, n, m, k, &alpha_f, b, CUDA_R_16F, ldb, a,
                                 CUDA_R_16F, lda, &beta_f, c, CUDA_R_16F, ldc, CUDA_R_32F,
                                 CUBLAS_GEMM_DFALT_TENSOR_OP));
  } else {
    OF_CUBLAS_CHECK(cublasSgemmEx(ctx->cublas_tensor_op_math_handle(), cublas_trans_b,
                                  cublas_trans_a, n, m, k, &alpha_f, b, CUDA_R_16F, ldb, a,
                                  CUDA_R_16F, lda, &beta_f, c, CUDA_R_16F, ldc));
  }
}

std::tuple<int, int, int> CalcMNKForGemm(enum CBLAS_TRANSPOSE trans_a, const Blob* a,
                                         const Blob* c) {
  const auto& a_shape = a->shape_view();
  const auto& c_shape = c->shape_view();
  int m = c_shape.At(0);
  int n = c_shape.Count(1);
  int k = (trans_a == CblasNoTrans) ? a_shape.Count(1) : a_shape.At(0);
  return std::make_tuple(m, n, k);
}

template<typename T>
void BlobGemmImpl(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                  T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
  int m, n, k;
  std::tie(m, n, k) = CalcMNKForGemm(trans_a, a, c);
  BlasIf<DeviceType::kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(),
                                   b->dptr<T>(), beta, c->mut_dptr<T>());
}

std::tuple<int, int, int, int, int, int, cublasOperation_t, cublasOperation_t>
PrepareToCallBatchedGemm(const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                         int batch_size, int m, int n, int k) {
  const int a_stride = m * k;
  const int b_stride = k * n;
  const int c_stride = m * n;
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  return std::make_tuple(a_stride, b_stride, c_stride, lda, ldb, ldc, cublas_trans_a,
                         cublas_trans_b);
}

template<typename T>
cudaDataType_t GetCudaDataType4BatchedGemm() {
  return CudaDataType<T>::value;
}

template<>
cudaDataType_t GetCudaDataType4BatchedGemm<half>() {
  return CUDA_R_16F;
}

template<typename T>
void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                     int batch_size, int m, int n, int k, const T* alpha, const T* a, const T* b,
                     const T* beta, T* c, T** buf) {
  int a_stride, b_stride, c_stride;
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(a_stride, b_stride, c_stride, lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallBatchedGemm(trans_a, trans_b, batch_size, m, n, k);

  if (CUDA_VERSION >= 9010 && GetCudaSmVersion() >= 500) {
#if CUDA_VERSION >= 9010
    cudaDataType_t data_type = GetCudaDataType4BatchedGemm<T>();
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
        reinterpret_cast<const void*>(alpha), reinterpret_cast<const void*>(b), data_type, ldb,
        b_stride, reinterpret_cast<const void*>(a), data_type, lda, a_stride,
        reinterpret_cast<const void*>(beta), reinterpret_cast<void*>(c), data_type, ldc, c_stride,
        batch_size, data_type, CUBLAS_GEMM_DEFAULT));
#endif
  } else {
    cublas_gemmStridedBatched<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                                 alpha, b, ldb, b_stride, a, lda, a_stride, beta, c, ldc, c_stride,
                                 batch_size);
  }
}

#if CUDA_VERSION >= 9010
template<>
void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                     int batch_size, int m, int n, int k, const half* alpha, const half* a,
                     const half* b, const half* beta, half* c, half** buf) {
  float alpha_f = __half2float(*alpha);
  float beta_f = __half2float(*beta);

  int a_stride, b_stride, c_stride;
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(a_stride, b_stride, c_stride, lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallBatchedGemm(trans_a, trans_b, batch_size, m, n, k);

  if (GetCudaSmVersion() >= 500) {
#if CUDA_VERSION >= 11000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        ctx->cublas_tensor_op_math_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha_f,
        reinterpret_cast<const void*>(b), CUDA_R_16F, ldb, b_stride,
        reinterpret_cast<const void*>(a), CUDA_R_16F, lda, a_stride, &beta_f,
        reinterpret_cast<void*>(c), CUDA_R_16F, ldc, c_stride, batch_size, CUDA_R_32F, algo));
  } else {
    cublas_gemmStridedBatched<half>(ctx->cublas_tensor_op_math_handle(), cublas_trans_b,
                                    cublas_trans_a, n, m, k, alpha, b, ldb, b_stride, a, lda,
                                    a_stride, beta, c, ldc, c_stride, batch_size);
  }
}
#endif

__global__ void AxpyHalfGpu(const int n, const half alpha, const half* x, const int incx, half* y,
                            const int incy) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i * incy] = __hfma(alpha, x[i * incx], y[i * incy]); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

}  // namespace

void BlasIf<DeviceType::kGPU>::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                        enum CBLAS_TRANSPOSE trans_b, float alpha, float beta,
                                        const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<float>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
void BlasIf<DeviceType::kGPU>::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                        enum CBLAS_TRANSPOSE trans_b, double alpha, double beta,
                                        const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<double>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
void BlasIf<DeviceType::kGPU>::BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                        enum CBLAS_TRANSPOSE trans_b, float16 alpha, float16 beta,
                                        const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<float16>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
void BlasIf<DeviceType::kGPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const float alpha, const float* a,
                                      const float* b, const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, &alpha, a, b, &beta, c);
}
void BlasIf<DeviceType::kGPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const double alpha, const double* a,
                                      const double* b, const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, &alpha, a, b, &beta, c);
}
void BlasIf<DeviceType::kGPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const float16 alpha, const float16* a,
                                      const float16* b, const float16 beta, float16* c) {
  Gemm<half>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, reinterpret_cast<const half*>(&alpha),
             reinterpret_cast<const half*>(a), reinterpret_cast<const half*>(b),
             reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c));
}

void BlasIf<DeviceType::kGPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const float alpha, const float* a, const float* b,
                                             const float beta, float* c, float** buf) {
  BatchedGemmImpl<float>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, &alpha, a, b,
                         &beta, c, buf);
}
void BlasIf<DeviceType::kGPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const double alpha, const double* a, const double* b,
                                             const double beta, double* c, double** buf) {
  BatchedGemmImpl<double>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, &alpha, a, b,
                          &beta, c, buf);
}
void BlasIf<DeviceType::kGPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const float16 alpha, const float16* a,
                                             const float16* b, const float16 beta, float16* c,
                                             float16** buf) {
  BatchedGemmImpl<half>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k,
                        reinterpret_cast<const half*>(&alpha), reinterpret_cast<const half*>(a),
                        reinterpret_cast<const half*>(b), reinterpret_cast<const half*>(&beta),
                        reinterpret_cast<half*>(c), reinterpret_cast<half**>(buf));
}

void BlasIf<DeviceType::kGPU>::Axpy(DeviceCtx* ctx, const int n, const float alpha, const float* x,
                                    const int incx, float* y, const int incy) {
  cublas_axpy<float>(ctx->cublas_pmh_handle(), n, &alpha, x, incx, y, incy);
}

void BlasIf<DeviceType::kGPU>::Axpy(DeviceCtx* ctx, const int n, const double alpha,
                                    const double* x, const int incx, double* y, const int incy) {
  cublas_axpy<double>(ctx->cublas_pmh_handle(), n, &alpha, x, incx, y, incy);
}

void BlasIf<DeviceType::kGPU>::Axpy(DeviceCtx* ctx, const int n, const float16 alpha,
                                    const float16* x, const int incx, float16* y, const int incy) {
  AxpyHalfGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, float16_2half(alpha), reinterpret_cast<const half*>(x), incx, reinterpret_cast<half*>(y),
      incy);
}

}  // namespace oneflow
