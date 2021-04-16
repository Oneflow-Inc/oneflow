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
#ifdef WITH_CUDA

#include "oneflow/core/cuda/bgemm.h"
#include "oneflow/core/common/blas.h"

namespace oneflow {
namespace cuda {
namespace blas {

namespace {

inline cublasOperation_t Char2CublasOp(char op) {
  switch (op) {
    case 'n':
    case 'N': {
      return CUBLAS_OP_N;
    }
    case 't':
    case 'T': {
      return CUBLAS_OP_T;
    }
    case 'c':
    case 'C': {
      return CUBLAS_OP_C;
    }
    default: {
      UNIMPLEMENTED();
    }
  }
  return CUBLAS_OP_N;
}

}  // namespace

template<>
void bgemm<float>(cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n, int64_t k,
                  float alpha, const float* a, int64_t lda, int64_t stridea, const float* b,
                  int64_t ldb, int64_t strideb, float beta, float* c, int64_t ldc, int64_t stridec,
                  int64_t batch_size) {
  cublasOperation_t opa = Char2CublasOp(transa);
  cublasOperation_t opb = Char2CublasOp(transb);
  if (CUDA_VERSION >= 9010 && GetCudaSmVersion() >= 500) {
#if CUDA_VERSION >= 9010
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, opa, opb, m, n, k, reinterpret_cast<const void*>(&alpha),
        reinterpret_cast<const void*>(a), CUDA_R_16F, lda, stridea,
        reinterpret_cast<const void*>(a), CUDA_R_16F, ldb, strideb,
        reinterpret_cast<const void*>(&beta), reinterpret_cast<void*>(c), CUDA_R_16F, ldc, stridec,
        batch_size, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
#else
    UNIMPLEMENTED();
#endif
  } else {
    cublas_gemmStridedBatched<float>(handle, opa, opb, m, n, k, &alpha, a, ldb, stridea, b, ldb,
                                     strideb, &beta, c, ldc, stridec, batch_size);
  }
}

#if CUDA_VERSION >= 9010

template<>
void bgemm<half>(cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n, int64_t k,
                 half alpha, const half* a, int64_t lda, int64_t stridea, const half* b,
                 int64_t ldb, int64_t strideb, half beta, half* c, int64_t ldc, int64_t stridec,
                 int64_t batch_size) {
  cublasOperation_t opa = Char2CublasOp(transa);
  cublasOperation_t opb = Char2CublasOp(transb);

  if (GetCudaSmVersion() >= 500) {
    float alpha_f = static_cast<float>(alpha);
    float beta_f = static_cast<float>(beta);
#if CUDA_VERSION >= 11000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, opa, opb, m, n, k, &alpha_f, reinterpret_cast<const void*>(a), CUDA_R_16F, lda,
        stridea, reinterpret_cast<const void*>(b), CUDA_R_16F, ldb, strideb, &beta_f,
        reinterpret_cast<void*>(c), CUDA_R_16F, ldc, stridec, batch_size, CUDA_R_32F, algo));
  } else {
    cublas_gemmStridedBatched<half>(handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb,
                                    strideb, &beta, c, ldc, stridec, batch_size);
  }
}

template<>
void bgemm<float16>(cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n,
                    int64_t k, float16 alpha, const float16* a, int64_t lda, int64_t stridea,
                    const float16* b, int64_t ldb, int64_t strideb, float16 beta, float16* c,
                    int64_t ldc, int64_t stridec, int64_t batch_size) {
  half alpha_h = static_cast<half>(alpha);
  half beta_h = static_cast<half>(beta);
  bgemm<half>(handle, transa, transb, m, n, k, alpha_h, reinterpret_cast<const half*>(a), lda,
              stridea, reinterpret_cast<const half*>(b), ldb, strideb, beta_h,
              reinterpret_cast<half*>(c), ldc, stridec, batch_size);
}

#endif  // CUDA_VERSION >= 9010

}  // namespace blas
}  // namespace cuda
}  // namespace oneflow

#endif  // WITH_CUDA
