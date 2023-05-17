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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <functional>
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

CublasScalarParameter GetCublasScalarParameter(Scalar scalar, cublasComputeType_t compute_type) {
  CublasScalarParameter sp{};
  if (compute_type == CUBLAS_COMPUTE_64F) {
    sp.d = scalar.Value<double>();
  } else if (compute_type == CUBLAS_COMPUTE_32F_PEDANTIC
             || compute_type == CUBLAS_COMPUTE_32F_FAST_TF32
             || compute_type == CUBLAS_COMPUTE_32F) {
    sp.s = scalar.Value<float>();
  } else if (compute_type == CUBLAS_COMPUTE_16F) {
    sp.h = static_cast<half>(scalar.Value<float>());
  } else {
    UNIMPLEMENTED();
  }
  return sp;
}

cudaDataType_t GetCublasScalarType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    default: return CUDA_R_32F;
  }
}

cublasComputeType_t GetComputeType(DataType data_type, bool use_lt_interface) {
  switch (data_type) {
    case kFloat: {
      const bool allow_tf32 = ParseBooleanFromEnv("ONEFLOW_ALLOW_TF32", false);
      if (allow_tf32) {
        return CUBLAS_COMPUTE_32F_FAST_TF32;
      } else {
        // Starting with cuBLAS version 11.0.0, the library will automatically make use of Tensor
        // Core capabilities wherever possible, unless they are explicitly disabled by selecting
        // pedantic compute modes in cuBLAS
        return use_lt_interface ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC;
      }
    }
    case kDouble: return CUBLAS_COMPUTE_64F;
    case kFloat16: {
      const bool allow_half_accumulation =
          ParseBooleanFromEnv("ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION", false);
      if (allow_half_accumulation) {
        return CUBLAS_COMPUTE_16F;
      } else {
        return CUBLAS_COMPUTE_32F;
      }
    }
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUBLAS_COMPUTE_32F;
#endif  // CUDA_VERSION >= 11000
    default: UNIMPLEMENTED(); return CUBLAS_COMPUTE_32F;
  }
}

template<typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) { OF_CUBLAS_CHECK(destructor(x)); }
  }
};

template<typename T>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const { return descriptor_.get(); }
  T* descriptor() { return descriptor_.get(); }

 protected:
  std::shared_ptr<T> descriptor_;
};

class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<cublasLtMatmulDescOpaque_t> {
 public:
  CuBlasLtMatmulDescriptor(cublasComputeType_t compute_type, cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_ = std::shared_ptr<cublasLtMatmulDescOpaque_t>(
        raw_descriptor, CuBlasLtDeleter<cublasLtMatmulDescOpaque_t, &cublasLtMatmulDescDestroy>{});
  }
};

class CuBlasLtMatrixLayout : public CuBlasLtDescriptor<cublasLtMatrixLayoutOpaque_t> {
 public:
  CuBlasLtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&raw_descriptor, type, rows, cols, ld));
    descriptor_ = std::shared_ptr<cublasLtMatrixLayoutOpaque_t>(
        raw_descriptor,
        CuBlasLtDeleter<cublasLtMatrixLayoutOpaque_t, &cublasLtMatrixLayoutDestroy>{});
  }
};

class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<cublasLtMatmulPreferenceOpaque_t> {
 public:
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    OF_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_ = std::shared_ptr<cublasLtMatmulPreferenceOpaque_t>(
        raw_descriptor,
        CuBlasLtDeleter<cublasLtMatmulPreferenceOpaque_t, &cublasLtMatmulPreferenceDestroy>{});
  }
};

auto LaunchBroadcastMatmulLt(CudaStream* cuda_stream, DataType data_type,
                             const cudaDataType_t cuda_data_type,
                             const cublasComputeType_t compute_type,
                             const cublasOperation_t cublas_trans_a,
                             const cublasOperation_t cublas_trans_b, const int64_t cublas_m,
                             const int64_t cublas_n, const int64_t cublas_k,
                             const int64_t cublas_lda, const int64_t cublas_ldb,
                             const int64_t cublas_ldc, const void* sp_alpha) {
  const auto scalar_type = GetCublasScalarType(data_type);
  CuBlasLtMatmulDescriptor compute_desc(compute_type, scalar_type);
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(compute_desc.descriptor(),
                                                 CUBLASLT_MATMUL_DESC_TRANSA, &cublas_trans_a,
                                                 sizeof(cublas_trans_a)));
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(compute_desc.descriptor(),
                                                 CUBLASLT_MATMUL_DESC_TRANSB, &cublas_trans_b,
                                                 sizeof(cublas_trans_b)));
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      compute_desc.descriptor(), CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  CuBlasLtMatrixLayout a_desc(cuda_data_type, (cublas_trans_a == CUBLAS_OP_T) ? cublas_k : cublas_m,
                              (cublas_trans_a == CUBLAS_OP_T) ? cublas_m : cublas_k, cublas_lda);
  CuBlasLtMatrixLayout b_desc(cuda_data_type, (cublas_trans_b == CUBLAS_OP_T) ? cublas_n : cublas_k,
                              (cublas_trans_b == CUBLAS_OP_T) ? cublas_k : cublas_n, cublas_ldb);
  CuBlasLtMatrixLayout c_desc(cuda_data_type, cublas_m, cublas_n, cublas_ldc);

  CuBlasLtMatmulPreference preference;

  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
  // setting this to 1M.
  size_t workspace_size = 0;
  OF_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference.descriptor(),
                                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspace_size, sizeof(workspace_size)));
  cublasLtHandle_t lt_handle = cuda_stream->cublas_lt_handle();

  void* workspace = nullptr;
  // CHECK_JUST(cuda_stream->AllocAsync(&workspace, workspace_size));

  auto lt_matmul_func = [preference](
                            cublasLtHandle_t lt_handle, CuBlasLtMatmulDescriptor compute_desc,
                            const void* sp_alpha, const void* cublas_a, CuBlasLtMatrixLayout a_desc,
                            const void* cublas_b, CuBlasLtMatrixLayout b_desc, const void* sp_beta,
                            void* cublas_c, CuBlasLtMatrixLayout c_desc, void* workspace,
                            size_t workspace_size, CudaStream* cuda_stream) -> void {
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        compute_desc.descriptor(), CUBLASLT_MATMUL_DESC_BIAS_POINTER, &cublas_c, sizeof(void*)));
    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    int ret = 0;
    OF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt_handle, compute_desc.descriptor(), a_desc.descriptor(), b_desc.descriptor(),
        c_desc.descriptor(), c_desc.descriptor(), preference.descriptor(), 1, &heuristic_result,
        &ret));
    if (ret == 0) { OF_CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED); }

    OF_CUBLAS_CHECK(cublasLtMatmul(lt_handle, compute_desc.descriptor(), sp_alpha, cublas_a,
                                   a_desc.descriptor(), cublas_b, b_desc.descriptor(), sp_beta,
                                   cublas_c, c_desc.descriptor(), cublas_c, c_desc.descriptor(),
                                   &heuristic_result.algo, nullptr, 0, cuda_stream->cuda_stream()));
    // CHECK_JUST(cuda_stream->FreeAsync(&workspace));
  };

  return std::bind(lt_matmul_func, lt_handle, compute_desc, sp_alpha, std::placeholders::_1, a_desc,
                   std::placeholders::_2, b_desc, std::placeholders::_3, std::placeholders::_4,
                   c_desc, workspace, workspace_size, cuda_stream);
}

auto LaunchBroadcastMatmulEx(CudaStream* cuda_stream, DataType data_type,
                             const cudaDataType_t cuda_data_type,
                             const cublasComputeType_t compute_type,
                             const cublasOperation_t cublas_trans_a,
                             const cublasOperation_t cublas_trans_b, const int64_t cublas_m,
                             const int64_t cublas_n, const int64_t cublas_k,
                             const int64_t cublas_lda, const int64_t cublas_ldb,
                             const int64_t cublas_ldc, const void* sp_alpha) {
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

  auto ex_matmul_func =
      [](cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
         int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
         cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
         cublasComputeType_t computeType, cublasGemmAlgo_t algo) -> void {
    OF_CUBLAS_CHECK(cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype,
                                 ldb, beta, C, Ctype, ldc, computeType, algo));
  };

  return std::bind(ex_matmul_func, cuda_stream->cublas_handle(), cublas_trans_a, cublas_trans_b,
                   cublas_m, cublas_n, cublas_k, sp_alpha, std::placeholders::_1, cuda_data_type,
                   cublas_lda, std::placeholders::_2, cuda_data_type, cublas_ldb,
                   std::placeholders::_3, std::placeholders::_4, cuda_data_type, cublas_ldc,
                   compute_type, algo);
}

void LaunchBroadcastMatmul(Stream* stream, DataType data_type, BlasTransposeType transpose_a,
                           BlasTransposeType transpose_b, int64_t num_batch_dims,
                           const int64_t* broadcast_batch_dims, const int64_t* a_batch_dims,
                           const int64_t* b_batch_dims, const int64_t* c_batch_dims, int64_t m,
                           int64_t n, int64_t k, Scalar alpha, const void* a, const void* b,
                           Scalar beta, void* c) {
  auto scalar_equal_one = [](Scalar alpha) -> bool {
    return (alpha.IsIntegral() && alpha.Value<int64_t>() == 1)
           || (alpha.IsFloatingPoint()
               && std::fabs(alpha.Value<double>() - 1.0) < std::numeric_limits<double>::epsilon());
  };

  bool use_lt_interface = false;
#if CUDA_VERSION >= 11040 && !defined(_MSC_VER)
  use_lt_interface = scalar_equal_one(beta);
#endif

  auto* cuda_stream = stream->As<CudaStream>();
  const auto cuda_data_type = GetCudaDataType(data_type);
  const auto compute_type = GetComputeType(data_type, use_lt_interface);
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

  std::function<void(const void* cublas_a, const void* cublas_b, const void* sp_beta,
                     void* cublas_c)>
      matmul_func;
  if (use_lt_interface) {
    matmul_func = LaunchBroadcastMatmulLt(cuda_stream, data_type, cuda_data_type, compute_type,
                                          cublas_trans_a, cublas_trans_b, cublas_m, cublas_n,
                                          cublas_k, cublas_lda, cublas_ldb, cublas_ldc, &sp_alpha);
  } else {
    matmul_func = LaunchBroadcastMatmulEx(cuda_stream, data_type, cuda_data_type, compute_type,
                                          cublas_trans_a, cublas_trans_b, cublas_m, cublas_n,
                                          cublas_k, cublas_lda, cublas_ldb, cublas_ldc, &sp_alpha);
  }

  if (num_batch_dims == 1 && c_batch_dims[0] != 1) {
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
      matmul_func(cublas_a, cublas_b, &sp_beta, cublas_c);
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
