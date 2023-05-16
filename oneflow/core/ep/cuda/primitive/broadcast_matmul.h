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

#ifndef ONEFLOW_CORE_EP_CUDA_BROADCAST_PRIMITIVE_MATMUL_H_
#define ONEFLOW_CORE_EP_CUDA_BROADCAST_PRIMITIVE_MATMUL_H_

#include "oneflow/core/device/cuda_util.h"

#ifdef WITH_CUDA

// cublasLT was introduced in CUDA 10.1 but we enable only for 11.1 that also
// added bf16 support
// #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && !defined(_MSC_VER)
#include <cublasLt.h>
// #endif

namespace oneflow {

namespace ep {

namespace primitive {

namespace broadcast_matmul {

namespace internal {

template<typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) { OF_CUBLAS_CHECK(destructor(x)); }
  }
};

template<typename T, cublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const { return descriptor_.get(); }
  T* descriptor() { return descriptor_.get(); }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatmulDescriptor
    : public CuBlasLtDescriptor<cublasLtMatmulDescOpaque_t, &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(cublasComputeType_t compute_type, cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
};

class CuBlasLtMatrixLayout
    : public CuBlasLtDescriptor<cublasLtMatrixLayoutOpaque_t, &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&raw_descriptor, type, rows, cols, ld));
    descriptor_.reset(raw_descriptor);
  }
};

class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<cublasLtMatmulPreferenceOpaque_t,
                                                           &cublasLtMatmulPreferenceDestroy> {
 public:
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    OF_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

}  // namespace internal

}  // namespace broadcast_matmul

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_EP_CUDA_BROADCAST_PRIMITIVE_MATMUL_H_