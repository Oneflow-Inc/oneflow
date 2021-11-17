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
  cublasOperation_t cublas_trans{};
  if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_N;
  } else if (trans == CBLAS_TRANSPOSE::CblasTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_T;
  } else if (trans == CBLAS_TRANSPOSE::CblasConjTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_C;
  } else {
    UNIMPLEMENTED();
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
void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER /*order*/, enum CBLAS_TRANSPOSE trans_a,
          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const double alpha,
          const T* a, const T* b, const double beta, T* c) {
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  cublasOperation_t cublas_trans_a{};
  cublasOperation_t cublas_trans_b{};
  std::tie(lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallCublasGemm(trans_a, trans_b, m, n, k);

  const T alpha_val = static_cast<T>(alpha);
  const T beta_val = static_cast<T>(beta);
  cublas_gemm<T>(ctx->cublas_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha_val, b, ldb,
                 a, lda, &beta_val, c, ldc);
}

template<>
void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER /*order*/, enum CBLAS_TRANSPOSE trans_a,
          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const double alpha,
          const half* a, const half* b, const double beta, half* c) {
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f = static_cast<float>(beta);
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  cublasOperation_t cublas_trans_a{};
  cublasOperation_t cublas_trans_b;
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

}  // namespace oneflow
