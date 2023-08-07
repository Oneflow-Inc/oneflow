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
#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"
#include "oneflow/core/ep/common/primitive/broadcast_matmul.h"
#include "oneflow/core/common/blas.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace broadcast_matmul {

namespace internal {

namespace {

constexpr size_t kMaxNumDims = 8;

CBLAS_TRANSPOSE GetCblasTranspose(BlasTransposeType transpose_type, DataType data_type) {
  if (transpose_type == BlasTransposeType::N) {
    return CblasNoTrans;
  } else if (transpose_type == BlasTransposeType::T) {
    return DType(data_type).is_complex() ? CblasConjTrans : CblasTrans;
  } else {
    UNIMPLEMENTED();
    return CblasNoTrans;
  }
}

template<typename T, typename std::enable_if<
                         !(std::is_same<T, std::complex<float>>::value
                           || std::is_same<T, std::complex<double>>::value)>::type* = nullptr>
void CblasMatmul(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b, int m, int n, int k, T alpha,
                 const T* a, const T* b, T beta, T* c) {
  int lda = 0;
  if (trans_a == CblasNoTrans) {
    lda = k;
  } else if (trans_a == CblasTrans || trans_a == CblasConjTrans) {
    lda = m;
  } else {
    UNIMPLEMENTED();
  }
  int ldb = 0;
  if (trans_b == CblasNoTrans) {
    ldb = n;
  } else if (trans_b == CblasTrans || trans_b == CblasConjTrans) {
    ldb = k;
  } else {
    UNIMPLEMENTED();
  }
  const int ldc = n;
  cblas_gemm<T>(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<typename T,
         typename std::enable_if<std::is_same<T, std::complex<float>>::value
                                 || std::is_same<T, std::complex<double>>::value>::type* = nullptr>
void CblasMatmul(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b, int m, int n, int k, T alpha,
                 const T* a, const T* b, T beta, T* c) {
  int lda = 0;
  if (trans_a == CblasNoTrans) {
    lda = k;
  } else if (trans_a == CblasTrans || trans_a == CblasConjTrans) {
    lda = m;
  } else {
    UNIMPLEMENTED();
  }
  int ldb = 0;
  if (trans_b == CblasNoTrans) {
    ldb = n;
  } else if (trans_b == CblasTrans || trans_b == CblasConjTrans) {
    ldb = k;
  } else {
    UNIMPLEMENTED();
  }
  const int ldc = n;
  cblas_gemm<T>(CblasRowMajor, trans_a, trans_b, m, n, k, reinterpret_cast<const void*>(&alpha),
                reinterpret_cast<const void*>(a), lda, reinterpret_cast<const void*>(b), ldb,
                reinterpret_cast<const void*>(&beta), reinterpret_cast<void*>(c), ldc);
}

template<typename T>
void LaunchCblasBroadcastMatmul(Stream* /*stream*/, DataType data_type,
                                BlasTransposeType transpose_a, BlasTransposeType transpose_b,
                                int64_t num_batch_dims, const int64_t* broadcast_batch_dims,
                                const int64_t* a_batch_dims, const int64_t* b_batch_dims,
                                const int64_t* c_batch_dims, int64_t m, int64_t n, int64_t k,
                                Scalar alpha, const void* a, const void* b, Scalar beta, void* c) {
  const CBLAS_TRANSPOSE cblas_trans_a = GetCblasTranspose(transpose_a, data_type);
  const CBLAS_TRANSPOSE cblas_trans_b = GetCblasTranspose(transpose_b, data_type);
  const T alpha_value = alpha.Value<T>();
  auto func = [&](const void* batch_a, const void* batch_b, void* batch_c, Scalar batch_beta) {
    const T beta_value = batch_beta.Value<T>();
    CblasMatmul<T>(cblas_trans_a, cblas_trans_b, m, n, k, alpha_value,
                   static_cast<const T*>(batch_a), static_cast<const T*>(batch_b), beta_value,
                   static_cast<T*>(batch_c));
  };
  ForEachMatmul<kMaxNumDims>(data_type, m, n, k, beta, num_batch_dims, broadcast_batch_dims,
                             a_batch_dims, b_batch_dims, c_batch_dims, a, b, c, func);
}

void LaunchBroadcastMatmul(Stream* stream, DataType data_type, BlasTransposeType transpose_a,
                           BlasTransposeType transpose_b, int64_t num_batch_dims,
                           const int64_t* broadcast_batch_dims, const int64_t* a_batch_dims,
                           const int64_t* b_batch_dims, const int64_t* c_batch_dims, int64_t m,
                           int64_t n, int64_t k, Scalar alpha, const void* a, const void* b,
                           Scalar beta, void* c) {
  if (data_type == DataType::kFloat) {
    LaunchCblasBroadcastMatmul<float>(stream, data_type, transpose_a, transpose_b, num_batch_dims,
                                      broadcast_batch_dims, a_batch_dims, b_batch_dims,
                                      c_batch_dims, m, n, k, alpha, a, b, beta, c);
  } else if (data_type == DataType::kDouble) {
    LaunchCblasBroadcastMatmul<double>(stream, data_type, transpose_a, transpose_b, num_batch_dims,
                                       broadcast_batch_dims, a_batch_dims, b_batch_dims,
                                       c_batch_dims, m, n, k, alpha, a, b, beta, c);
  } else if (data_type == DataType::kComplex64) {
    LaunchCblasBroadcastMatmul<std::complex<float>>(
        stream, data_type, transpose_a, transpose_b, num_batch_dims, broadcast_batch_dims,
        a_batch_dims, b_batch_dims, c_batch_dims, m, n, k, alpha, a, b, beta, c);
  } else if (data_type == DataType::kComplex128) {
    LaunchCblasBroadcastMatmul<std::complex<double>>(
        stream, data_type, transpose_a, transpose_b, num_batch_dims, broadcast_batch_dims,
        a_batch_dims, b_batch_dims, c_batch_dims, m, n, k, alpha, a, b, beta, c);
  } else {
    UNIMPLEMENTED();
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
    if (max_num_dims > kMaxNumDims) { return nullptr; }
    if (data_type == DataType::kFloat || data_type == DataType::kDouble
        || data_type == DataType::kComplex64 || data_type == DataType::kComplex128) {
      return std::make_unique<BroadcastMatmulImpl<kMaxNumDims>>(data_type, transpose_a,
                                                                transpose_b);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, BroadcastMatmulFactory, BroadcastMatmulFactoryImpl);

}  // namespace

}  // namespace internal

}  // namespace broadcast_matmul

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
