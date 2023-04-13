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

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"
#include "oneflow/core/ep/common/primitive/broadcast_matmul.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>

namespace oneflow {

namespace ep {
namespace primitive {

namespace broadcast_matmul {

namespace internal {

namespace {

constexpr size_t kMaxNumDims = 8;

Optional<cudaDataType_t> OptCudaDataType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    case kFloat16: return CUDA_R_16F;
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUDA_R_16BF;
#endif  // CUDA_VERSION >= 11000
    default: return NullOpt;
  }
}

cudaDataType_t GetCudaDataType(DataType data_type) {
  auto cuda_data_type = OptCudaDataType(data_type);
  CHECK(cuda_data_type.has_value());
  return cuda_data_type.value_or(CUDA_R_32F);
}

union CublasScalarParameter {
  double d;
  float s;
  half h;
};

CublasScalarParameter GetCublasScalarParameter(Scalar scalar, cudaDataType_t compute_type) {
  CublasScalarParameter sp{};
  if (compute_type == CUDA_R_64F) {
    sp.d = scalar.Value<double>();
  } else if (compute_type == CUDA_R_32F) {
    sp.s = scalar.Value<float>();
  } else if (compute_type == CUDA_R_16F) {
    sp.h = static_cast<half>(scalar.Value<float>());
  } else {
    UNIMPLEMENTED();
  }
  return sp;
}

cudaDataType_t GetComputeType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    case kFloat16: {
      const bool allow_half_accumulation =
          ParseBooleanFromEnv("ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION", false);
      if (allow_half_accumulation) {
        return CUDA_R_16F;
      } else {
        return CUDA_R_32F;
      }
    }
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUDA_R_32F;
#endif  // CUDA_VERSION >= 11000
    default: UNIMPLEMENTED(); return CUDA_R_32F;
  }
}

void LaunchBroadcastMatmul(Stream* stream, DataType data_type, BlasTransposeType transpose_a,
                           BlasTransposeType transpose_b, int64_t num_batch_dims,
                           const int64_t* broadcast_batch_dims, const int64_t* a_batch_dims,
                           const int64_t* b_batch_dims, const int64_t* c_batch_dims, int64_t m,
                           int64_t n, int64_t k, Scalar alpha, const void* a, const void* b,
                           Scalar beta, void* c) {
  auto* cuda_stream = stream->As<CudaStream>();
  const auto cuda_data_type = GetCudaDataType(data_type);
  const auto compute_type = GetComputeType(data_type);
  const auto sp_alpha = GetCublasScalarParameter(alpha, compute_type);
  const auto GetCublasOperation = [](BlasTransposeType transpose_type) {
    if (transpose_type == BlasTransposeType::N) {
      return CUBLAS_OP_N;
    } else if (transpose_type == BlasTransposeType::T) {
      return CUBLAS_OP_T;
    } else {
      UNIMPLEMENTED();
      return CUBLAS_OP_N;
    }
  };
  const cublasOperation_t cublas_trans_a = GetCublasOperation(transpose_b);
  const cublasOperation_t cublas_trans_b = GetCublasOperation(transpose_a);
  const int cublas_m = n;
  const int cublas_n = m;
  const int cublas_k = k;
  int cublas_lda = 0;
  if (transpose_b == BlasTransposeType::N) {
    cublas_lda = n;
  } else if (transpose_b == BlasTransposeType::T) {
    cublas_lda = k;
  } else {
    UNIMPLEMENTED();
  }
  int cublas_ldb = 0;
  if (transpose_a == BlasTransposeType::N) {
    cublas_ldb = k;
  } else if (transpose_a == BlasTransposeType::T) {
    cublas_ldb = m;
  } else {
    UNIMPLEMENTED();
  }
  const int cublas_ldc = n;
  CublasMathModeGuard guard(cuda_stream->cublas_handle());
  if (data_type == DataType::kFloat16) {
#if CUDA_VERSION < 11000
    guard.SetMathMode(CUBLAS_TENSOR_OP_MATH);
#else
    guard.SetMathMode(CUBLAS_DEFAULT_MATH);
#endif  // CUDA_VERSION < 11000
  }
#if CUDA_VERSION >= 11000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
  cublasGemmAlgo_t algo =
      (data_type == DataType::kFloat16) ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
#endif
  if (num_batch_dims == 1 && c_batch_dims[0] != 1) {
    const void* cublas_a = b;
    const void* cublas_b = a;
    void* cublas_c = c;
    const int64_t a_batch_count = a_batch_dims[0];
    const int64_t b_batch_count = b_batch_dims[0];
    CHECK(a_batch_count == 1 || b_batch_count == 1 || a_batch_count == b_batch_count);
    CHECK_GT(a_batch_count, 0);
    CHECK_GT(b_batch_count, 0);
    const int batch_count = std::max(a_batch_count, b_batch_count);
    const long long int cublas_stride_a = b_batch_count == 1 ? 0 : cublas_m * cublas_k;
    const long long int cublas_stride_b = a_batch_count == 1 ? 0 : cublas_k * cublas_n;
    const long long int cublas_stride_c = cublas_m * cublas_n;
    const auto sp_beta = GetCublasScalarParameter(beta, compute_type);
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cuda_stream->cublas_handle(), cublas_trans_a, cublas_trans_b, cublas_m, cublas_n, cublas_k,
        &sp_alpha, cublas_a, cuda_data_type, cublas_lda, cublas_stride_a, cublas_b, cuda_data_type,
        cublas_ldb, cublas_stride_b, &sp_beta, cublas_c, cuda_data_type, cublas_ldc,
        cublas_stride_c, batch_count, compute_type, algo));
  } else {
    auto func = [&](const void* batch_a, const void* batch_b, void* batch_c, Scalar batch_beta) {
      const auto sp_beta = GetCublasScalarParameter(batch_beta, compute_type);
      const void* cublas_a = batch_b;
      const void* cublas_b = batch_a;
      void* cublas_c = batch_c;
      OF_CUBLAS_CHECK(cublasGemmEx(
          cuda_stream->cublas_handle(), cublas_trans_a, cublas_trans_b, cublas_m, cublas_n,
          cublas_k, &sp_alpha, cublas_a, cuda_data_type, cublas_lda, cublas_b, cuda_data_type,
          cublas_ldb, &sp_beta, cublas_c, cuda_data_type, cublas_ldc, compute_type, algo));
    };
    ForEachMatmul<kMaxNumDims>(data_type, m, n, k, beta, num_batch_dims, broadcast_batch_dims,
                               a_batch_dims, b_batch_dims, c_batch_dims, a, b, c, func);
  }
}

class BroadcastMatmulFactoryImpl : public BroadcastMatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMatmulFactoryImpl);
  BroadcastMatmulFactoryImpl() = default;
  ~BroadcastMatmulFactoryImpl() override = default;

  std::unique_ptr<BroadcastMatmul> New(DataType data_type, BlasTransposeType transpose_a,
                                       BlasTransposeType transpose_b,
                                       size_t max_num_dims) override {
    auto cuda_data_type = OptCudaDataType(data_type);
    if (max_num_dims <= kMaxNumDims && cuda_data_type.has_value()) {
      return std::make_unique<BroadcastMatmulImpl<kMaxNumDims>>(data_type, transpose_a,
                                                                transpose_b);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, BroadcastMatmulFactory, BroadcastMatmulFactoryImpl);

}  // namespace

}  // namespace internal

}  // namespace broadcast_matmul

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
