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
          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const double alpha,
          const T* a, const T* b, const double beta, T* c) {
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallCublasGemm(trans_a, trans_b, m, n, k);

  const T alpha_val = static_cast<T>(alpha);
  const T beta_val = static_cast<T>(beta);
  cublas_gemm<T>(ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha_val, b, ldb,
                 a, lda, &beta_val, c, ldc);
}

template<>
void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const double alpha,
          const half* a, const half* b, const double beta, half* c) {
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f = static_cast<float>(beta);
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallCublasGemm(trans_a, trans_b, m, n, k);
#if CUDA_VERSION < 11000
  CublasMathModeGuard guard(ctx->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
#else
  CublasMathModeGuard guard(ctx->cublas_handle(), CUBLAS_DEFAULT_MATH);
#endif  // CUDA_VERSION < 11000
  if (GetCudaSmVersion() >= 500) {
    OF_CUBLAS_CHECK(cublasGemmEx(ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                                 &alpha_f, b, CUDA_R_16F, ldb, a, CUDA_R_16F, lda, &beta_f, c,
                                 CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
  } else {
    OF_CUBLAS_CHECK(cublasSgemmEx(ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                                  &alpha_f, b, CUDA_R_16F, ldb, a, CUDA_R_16F, lda, &beta_f, c,
                                  CUDA_R_16F, ldc));
  }
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

#define CUDA_DATA_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(float, CUDA_R_32F)  \
  OF_PP_MAKE_TUPLE_SEQ(double, CUDA_R_64F) \
  OF_PP_MAKE_TUPLE_SEQ(float16, CUDA_R_16F)

template<typename T>
struct CudaDataType;

#define SPECIALIZE_CUDA_DATA_TYPE(type_cpp, type_cuda) \
  template<>                                           \
  struct CudaDataType<type_cpp> : std::integral_constant<cudaDataType_t, type_cuda> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_CUDA_DATA_TYPE, CUDA_DATA_TYPE_SEQ);
#undef SPECIALIZE_CUDA_DATA_TYPE

template<typename T>
cudaDataType_t GetCudaDataType4BatchedGemm() {
  return CudaDataType<T>::value;
}

template<typename T>
void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                     int batch_size, int m, int n, int k, const double alpha, const T* a,
                     const T* b, const double beta, T* c) {
  int a_stride, b_stride, c_stride;
  int lda, ldb, ldc;
  const T alpha_val = static_cast<T>(alpha);
  const T beta_val = static_cast<T>(beta);
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(a_stride, b_stride, c_stride, lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallBatchedGemm(trans_a, trans_b, batch_size, m, n, k);

  if (CUDA_VERSION >= 9010 && GetCudaSmVersion() >= 500) {
#if CUDA_VERSION >= 9010
    cudaDataType_t data_type = GetCudaDataType4BatchedGemm<T>();
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
        reinterpret_cast<const void*>(&alpha_val), reinterpret_cast<const void*>(b), data_type, ldb,
        b_stride, reinterpret_cast<const void*>(a), data_type, lda, a_stride,
        reinterpret_cast<const void*>(&beta_val), reinterpret_cast<void*>(c), data_type, ldc,
        c_stride, batch_size, data_type, CUBLAS_GEMM_DEFAULT));
#endif
  } else {
    cublas_gemmStridedBatched<T>(ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                                 &alpha_val, b, ldb, b_stride, a, lda, a_stride, &beta_val, c, ldc,
                                 c_stride, batch_size);
  }
}

#if CUDA_VERSION >= 9010
template<>
void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                     int batch_size, int m, int n, int k, const double alpha, const half* a,
                     const half* b, const double beta, half* c) {
  int a_stride, b_stride, c_stride;
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(a_stride, b_stride, c_stride, lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallBatchedGemm(trans_a, trans_b, batch_size, m, n, k);
#if CUDA_VERSION < 11000
  CublasMathModeGuard guard(ctx->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
#else
  CublasMathModeGuard guard(ctx->cublas_handle(), CUBLAS_DEFAULT_MATH);
#endif  // CUDA_VERSION < 11000
  if (GetCudaSmVersion() >= 500) {
    const float alpha_f = static_cast<float>(alpha);
    const float beta_f = static_cast<float>(beta);
#if CUDA_VERSION >= 11000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha_f,
        reinterpret_cast<const void*>(b), CUDA_R_16F, ldb, b_stride,
        reinterpret_cast<const void*>(a), CUDA_R_16F, lda, a_stride, &beta_f,
        reinterpret_cast<void*>(c), CUDA_R_16F, ldc, c_stride, batch_size, CUDA_R_32F, algo));
  } else {
    const half alpha_h = static_cast<half>(alpha);
    const half beta_h = static_cast<half>(beta);
    cublas_gemmStridedBatched<half>(ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                                    &alpha_h, b, ldb, b_stride, a, lda, a_stride, &beta_h, c, ldc,
                                    c_stride, batch_size);
  }
}
#endif

}  // namespace

void BlasIf<DeviceType::kGPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const double alpha, const float* a,
                                      const float* b, const double beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
void BlasIf<DeviceType::kGPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const double alpha, const double* a,
                                      const double* b, const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
void BlasIf<DeviceType::kGPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const double alpha, const float16* a,
                                      const float16* b, const double beta, float16* c) {
  Gemm<half>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, reinterpret_cast<const half*>(a),
             reinterpret_cast<const half*>(b), beta, reinterpret_cast<half*>(c));
}

void BlasIf<DeviceType::kGPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const double alpha, const float* a, const float* b,
                                             const double beta, float* c) {
  BatchedGemmImpl<float>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                         beta, c);
}
void BlasIf<DeviceType::kGPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const double alpha, const double* a, const double* b,
                                             const double beta, double* c) {
  BatchedGemmImpl<double>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                          beta, c);
}
void BlasIf<DeviceType::kGPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const double alpha, const float16* a, const float16* b,
                                             const double beta, float16* c) {
  BatchedGemmImpl<half>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha,
                        reinterpret_cast<const half*>(a), reinterpret_cast<const half*>(b), beta,
                        reinterpret_cast<half*>(c));
}

}  // namespace oneflow
