/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
*/
#pragma once
#include "oneflow/core/device/cuda_to_hip.h"
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <cstdint>
#include <utility>

// cuBLAS keeps C++ overloads of cublasGemmEx, cublasGemmBatchedEx and cublasGemmStridedBatchedEx
// that take the compute type as a data type (the signatures from before cublasComputeType_t
// existed). hipBLAS has only the compute-type forms, so map the data type the same way cuBLAS does.
inline bool OfHipblasComputeType(hipDataType data_type, hipblasComputeType_t* compute_type) {
  switch (data_type) {
    case HIP_R_16F: *compute_type = HIPBLAS_COMPUTE_16F; return true;
    case HIP_R_32F: *compute_type = HIPBLAS_COMPUTE_32F; return true;
    case HIP_R_64F: *compute_type = HIPBLAS_COMPUTE_64F; return true;
    case HIP_R_32I: *compute_type = HIPBLAS_COMPUTE_32I; return true;
    default: return false;
  }
}

inline hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle, hipblasOperation_t transA,
                                     hipblasOperation_t transB, int m, int n, int k,
                                     const void* alpha, const void* A, hipDataType aType, int lda,
                                     const void* B, hipDataType bType, int ldb, const void* beta,
                                     void* C, hipDataType cType, int ldc, hipDataType computeType,
                                     hipblasGemmAlgo_t algo) {
  hipblasComputeType_t compute_type;
  if (!OfHipblasComputeType(computeType, &compute_type)) { return HIPBLAS_STATUS_INVALID_ENUM; }
  return hipblasGemmEx(handle, transA, transB, m, n, k, alpha, A, aType, lda, B, bType, ldb, beta,
                       C, cType, ldc, compute_type, algo);
}

// The pointer arrays follow cuBLAS's `const void* const[]`, which also accepts `void**`.
inline hipblasStatus_t hipblasGemmBatchedEx(
    hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n,
    int k, const void* alpha, const void* const A[], hipDataType aType, int lda,
    const void* const B[], hipDataType bType, int ldb, const void* beta, void* const C[],
    hipDataType cType, int ldc, int batchCount, hipDataType computeType, hipblasGemmAlgo_t algo) {
  hipblasComputeType_t compute_type;
  if (!OfHipblasComputeType(computeType, &compute_type)) { return HIPBLAS_STATUS_INVALID_ENUM; }
  return hipblasGemmBatchedEx(handle, transA, transB, m, n, k, alpha, const_cast<const void**>(A),
                              aType, lda, const_cast<const void**>(B), bType, ldb, beta,
                              const_cast<void**>(C), cType, ldc, batchCount, compute_type, algo);
}

inline hipblasStatus_t hipblasGemmStridedBatchedEx(
    hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n,
    int k, const void* alpha, const void* A, hipDataType aType, int lda, hipblasStride strideA,
    const void* B, hipDataType bType, int ldb, hipblasStride strideB, const void* beta, void* C,
    hipDataType cType, int ldc, hipblasStride strideC, int batchCount, hipDataType computeType,
    hipblasGemmAlgo_t algo) {
  hipblasComputeType_t compute_type;
  if (!OfHipblasComputeType(computeType, &compute_type)) { return HIPBLAS_STATUS_INVALID_ENUM; }
  return hipblasGemmStridedBatchedEx(handle, transA, transB, m, n, k, alpha, A, aType, lda, strideA,
                                     B, bType, ldb, strideB, beta, C, cType, ldc, strideC,
                                     batchCount, compute_type, algo);
}

// rocBLAS gemm_ex splits k into chunks once ld * k of either operand reaches 2^30 and adds each
// chunk's product into D, so a 16-bit D is rounded once per chunk (about 1025 times for k = 2^20)
// and the error grows with k. hipBLASLt accumulates the whole product in float and rounds once
// (it is also much faster on these shapes), so callers send such GEMMs to OfLongKGemm instead.
// On GPUs where OfMatrixCoreAccumulationDrifts holds, a float accumulator chain longer than
// kOfLongKGemmChain also loses precision in either library, so any longer k goes there too.
constexpr int64_t kOfLongKGemmChain = int64_t{1} << 15;
bool OfMatrixCoreAccumulationDrifts();
inline bool OfUseLongKGemm(hipDataType d_type, int64_t k, int64_t lda, int64_t ldb) {
  return (d_type == HIP_R_16F || d_type == HIP_R_16BF)
         && (std::max(lda, ldb) * k >= (int64_t{1} << 30)
             || (k > kOfLongKGemmChain && OfMatrixCoreAccumulationDrifts()));
}

// Column-major strided batched D = alpha * op(A) * op(B) + beta * D with float compute, A and B of
// the given type and D of d_type.
inline hipblasStatus_t OfHipblasLtGemm(hipblasLtHandle_t handle, hipStream_t stream,
                                       void* workspace, size_t workspace_size,
                                       hipblasOperation_t trans_a, hipblasOperation_t trans_b,
                                       int64_t m, int64_t n, int64_t k, float alpha, const void* a,
                                       int64_t lda, int64_t stride_a, const void* b, int64_t ldb,
                                       int64_t stride_b, float beta, void* d, hipDataType type,
                                       hipDataType d_type, int64_t ldd, int64_t stride_d,
                                       int64_t batch_count) {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t layout_a = nullptr;
  hipblasLtMatrixLayout_t layout_b = nullptr;
  hipblasLtMatrixLayout_t layout_d = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  const bool ta = trans_a != HIPBLAS_OP_N;
  const bool tb = trans_b != HIPBLAS_OP_N;
  const int32_t batch = static_cast<int32_t>(batch_count);
  const uint64_t max_workspace = workspace_size;
  hipblasLtMatmulHeuristicResult_t heuristic{};
  int count = 0;
  hipblasStatus_t status = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  const auto Step = [&status](auto&& f) {
    if (status == HIPBLAS_STATUS_SUCCESS) { status = f(); }
  };
  Step([&] {
    return hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                           sizeof(trans_a));
  });
  Step([&] {
    return hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                           sizeof(trans_b));
  });
  Step([&] { return hipblasLtMatrixLayoutCreate(&layout_a, type, ta ? k : m, ta ? m : k, lda); });
  Step([&] { return hipblasLtMatrixLayoutCreate(&layout_b, type, tb ? n : k, tb ? k : n, ldb); });
  Step([&] { return hipblasLtMatrixLayoutCreate(&layout_d, d_type, m, n, ldd); });
  const std::pair<hipblasLtMatrixLayout_t, int64_t> strided[] = {
      {layout_a, stride_a}, {layout_b, stride_b}, {layout_d, stride_d}};
  for (const auto& layout : strided) {
    if (batch_count == 1) { break; }
    Step([&] {
      return hipblasLtMatrixLayoutSetAttribute(layout.first, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                               &batch, sizeof(batch));
    });
    Step([&] {
      return hipblasLtMatrixLayoutSetAttribute(layout.first,
                                               HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                               &layout.second, sizeof(layout.second));
    });
  }
  Step([&] { return hipblasLtMatmulPreferenceCreate(&pref); });
  Step([&] {
    return hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &max_workspace, sizeof(max_workspace));
  });
  Step([&] {
    const hipblasStatus_t found = hipblasLtMatmulAlgoGetHeuristic(
        handle, desc, layout_a, layout_b, layout_d, layout_d, pref, 1, &heuristic, &count);
    return found != HIPBLAS_STATUS_SUCCESS || count == 0 ? HIPBLAS_STATUS_NOT_SUPPORTED
                                                         : HIPBLAS_STATUS_SUCCESS;
  });
  Step([&] {
    return hipblasLtMatmul(handle, desc, &alpha, a, layout_a, b, layout_b, &beta, d, layout_d, d,
                           layout_d, &heuristic.algo, workspace, workspace_size, stream);
  });
  if (pref != nullptr) { hipblasLtMatmulPreferenceDestroy(pref); }
  if (layout_d != nullptr) { hipblasLtMatrixLayoutDestroy(layout_d); }
  if (layout_b != nullptr) { hipblasLtMatrixLayoutDestroy(layout_b); }
  if (layout_a != nullptr) { hipblasLtMatrixLayoutDestroy(layout_a); }
  if (desc != nullptr) { hipblasLtMatmulDescDestroy(desc); }
  return status;
}

// The GEMM OfUseLongKGemm selects: OfHipblasLtGemm, or where the accumulator drifts, k-chunks of
// at most kOfLongKGemmChain whose 16-bit partial products are summed in float; or, when hipBLASLt
// has no algorithm for a shape or tile (or ONEFLOW_EP_CUDA_LONG_K_GEMM_USE_HIPBLASLT=0), rocBLAS
// with a float D in tiles. Either way the 16-bit D is rounded from float once.
hipblasStatus_t OfLongKGemm(hipblasHandle_t handle, hipblasLtHandle_t lt_handle, hipStream_t stream,
                            void* workspace, size_t workspace_size, hipblasOperation_t trans_a,
                            hipblasOperation_t trans_b, int64_t m, int64_t n, int64_t k,
                            float alpha, const void* a, int64_t lda, int64_t stride_a,
                            const void* b, int64_t ldb, int64_t stride_b, float beta, void* d,
                            hipDataType type, int64_t ldd, int64_t stride_d, int64_t batch_count);
