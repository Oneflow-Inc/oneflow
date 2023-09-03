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
#ifdef WITH_CUTLASS

#ifndef ONEFLOW_USER_KERNELS_CUTLASS_GEMM_OPERATION_CACHE_KEY_H_
#define ONEFLOW_USER_KERNELS_CUTLASS_GEMM_OPERATION_CACHE_KEY_H_

#include "oneflow/core/framework/framework.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>

#ifdef WITH_CUTLASS_EXTENSION
#include <cutlass/library/cutlass_extension_library.h>
#endif  // WITH_CUTLASS_EXTENSION

namespace oneflow {

struct GemmOperationCacheKey {
  cutlass::library::GemmFunctionalKey functional_key;
  cutlass::library::GemmConfiguration configuraion;
  size_t alignment;
  bool fuse_scale_bias;
  bool fuse_residual;
  GemmOperationCacheKey(const cutlass::library::GemmFunctionalKey& functional_key,
                        const cutlass::library::GemmConfiguration& configuraion,
                        const cutlass::library::GemmArguments& arguments)
      : functional_key(functional_key),
        configuraion(configuraion),
        fuse_scale_bias(false),
        fuse_residual(false) {
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.A) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.B) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.C) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.D) % kCudaAlignSize, 0);
    const auto IsAligned = [&](size_t n) {
      return configuraion.lda % n == 0 && configuraion.ldb % n == 0 && configuraion.ldc % n == 0
             && configuraion.ldd % n == 0;
    };
    alignment = 128 / cutlass::library::sizeof_bits(functional_key.element_A);
    for (; alignment > 1; alignment = alignment >> 1) {
      if (IsAligned(alignment)) { break; }
    }
  }

#ifdef WITH_CUTLASS_EXTENSION
  GemmOperationCacheKey(cutlass::library::GemmFunctionalKey functional_key,
                        const cutlass::library::GemmScaleBiasFusionConfiguration& config,
                        const cutlass::library::GemmScaleBiasFusionArguments& arguments)
      : functional_key(functional_key),
        fuse_scale_bias(true),
        fuse_residual(arguments.Residual != nullptr) {
    configuraion.problem_size = config.problem_size;
    configuraion.split_k_slices = config.split_k_slices;
    configuraion.lda = config.lda;
    configuraion.ldb = config.ldb;
    configuraion.ldc = 0;
    configuraion.ldd = config.ldd;
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.A) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.B) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.Scale) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.Bias) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.Residual) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.D) % kCudaAlignSize, 0);
    const auto IsAligned = [&](size_t n) {
      return configuraion.lda % n == 0 && configuraion.ldb % n == 0 && config.ldr % n == 0
             && configuraion.ldd % n == 0;
    };
    alignment = 128 / cutlass::library::sizeof_bits(functional_key.element_A);
    for (; alignment > 1; alignment = alignment >> 1) {
      if (IsAligned(alignment)) { break; }
    }
  }
#endif  // WITH_CUTLASS_EXTENSION
};

struct GemmProblemSizeHasher {
  size_t operator()(const cutlass::gemm::GemmCoord& problem_size) const {
    size_t hash = 0;
    hash = HashCombine(hash, std::hash<int>()(problem_size.m()));
    hash = HashCombine(hash, std::hash<int>()(problem_size.n()));
    hash = HashCombine(hash, std::hash<int>()(problem_size.k()));
    return hash;
  }
};

struct GemmConfigurationHasher {
  size_t operator()(const cutlass::library::GemmConfiguration& configuraion) const {
    size_t hash = std::hash<int>()(configuraion.split_k_slices);
    hash = HashCombine(hash, GemmProblemSizeHasher()(configuraion.problem_size));
    hash = HashCombine(hash, configuraion.lda);
    hash = HashCombine(hash, configuraion.ldb);
    hash = HashCombine(hash, configuraion.ldc);
    hash = HashCombine(hash, configuraion.ldd);
    return hash;
  }
};

struct GemmOperationCacheKeyHasher {
  size_t operator()(const GemmOperationCacheKey& key) const {
    size_t hash = cutlass::library::GemmFunctionalKeyHasher()(key.functional_key);
    hash = HashCombine(hash, GemmConfigurationHasher()(key.configuraion));
    hash = HashCombine(hash, std::hash<size_t>()(key.alignment));
    hash = HashCombine(hash, std::hash<size_t>()(key.fuse_scale_bias));
    hash = HashCombine(hash, std::hash<size_t>()(key.fuse_residual));
    return hash;
  }
};

inline bool operator==(const cutlass::library::GemmConfiguration& lhs,
                       const cutlass::library::GemmConfiguration& rhs) {
  return lhs.split_k_slices == rhs.split_k_slices && lhs.problem_size == rhs.problem_size
         && lhs.lda == rhs.lda && lhs.ldb == rhs.ldb && lhs.ldc == rhs.ldc && lhs.ldd == rhs.ldd;
}

inline bool operator==(const GemmOperationCacheKey& lhs, const GemmOperationCacheKey& rhs) {
  return lhs.functional_key == rhs.functional_key && lhs.configuraion == rhs.configuraion
         && lhs.alignment == rhs.alignment && lhs.fuse_scale_bias == rhs.fuse_scale_bias
         && lhs.fuse_residual == rhs.fuse_residual;
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUTLASS_GEMM_OPERATION_CACHE_KEY_H_

#endif  // WITH_CUTLASS
