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

#ifndef ONEFLOW_USER_KERNELS_CUTLASS_GEMM_ARRAY_OPERATION_CACHE_KEY_H_
#define ONEFLOW_USER_KERNELS_CUTLASS_GEMM_ARRAY_OPERATION_CACHE_KEY_H_

#include "oneflow/core/framework/framework.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>

#ifdef WITH_CUTLASS_EXTENSION
#include <cutlass/library/cutlass_extension_singleton.h>
#include <cutlass/library/cutlass_extension_library.h>
#endif  // WITH_CUTLASS_EXTENSION

namespace oneflow {

struct GemmArrayOperationCacheKey {
  cutlass::library::GemmFunctionalKey functional_key;
  cutlass::library::GemmArrayConfiguration configuraion;
  size_t alignment;
  size_t kind;

  GemmArrayOperationCacheKey(const cutlass::library::GemmFunctionalKey& functional_key,
                             const cutlass::library::GemmArrayConfiguration& configuraion,
                             const cutlass::library::GemmArrayArguments& arguments)
      : functional_key(functional_key), configuraion(configuraion), kind(-1) {
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
  GemmArrayOperationCacheKey(cutlass::library::GemmFunctionalKey functional_key,
                             const cutlass::library::GemmArrayScaleBiasFusionConfiguration& config,
                             const cutlass::library::GemmArrayScaleBiasFusionArguments& arguments)
      : functional_key(functional_key) {
    if (arguments.Scale) {
      kind = arguments.Residual ? cutlass::library::SingletonKind::kGemmArrayScaleBiasResidualFusion
                                : cutlass::library::SingletonKind::kGemmArrayScaleBiasFusion;
    } else if (arguments.FilterScale) {
      if (arguments.Bias) {
        kind = arguments.Residual
                   ? cutlass::library::SingletonKind::kGemmArrayFilterScaleBiasResidualFusion
                   : cutlass::library::SingletonKind::kGemmArrayFilterScaleBiasFusion;
      } else {
        kind = arguments.Residual
                   ? cutlass::library::SingletonKind::kGemmArrayFilterScaleResidualFusion
                   : cutlass::library::SingletonKind::kGemmArrayFilterScaleFusion;
      }
    } else {
      UNIMPLEMENTED();
    }
    configuraion.problem_size = config.problem_size;
    configuraion.batch_count = config.batch_count;
    configuraion.lda = config.lda;
    configuraion.ldb = config.ldb;
    configuraion.ldc = 0;
    configuraion.ldd = config.ldd;
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

struct GemmArrayConfigurationHasher {
  size_t operator()(const cutlass::library::GemmArrayConfiguration& configuraion) const {
    size_t hash = 0;
    hash = HashCombine(hash, std::hash<int>()(configuraion.problem_size.m()));
    hash = HashCombine(hash, std::hash<int>()(configuraion.problem_size.n()));
    hash = HashCombine(hash, std::hash<int>()(configuraion.problem_size.k()));
    hash = HashCombine(hash, std::hash<int>()(configuraion.batch_count));
    hash = HashCombine(hash, configuraion.lda);
    hash = HashCombine(hash, configuraion.ldb);
    hash = HashCombine(hash, configuraion.ldc);
    hash = HashCombine(hash, configuraion.ldd);
    return hash;
  }
};

struct GemmArrayOperationCacheKeyHasher {
  size_t operator()(const GemmArrayOperationCacheKey& key) const {
    size_t hash = cutlass::library::GemmFunctionalKeyHasher()(key.functional_key);
    hash = HashCombine(hash, GemmArrayConfigurationHasher()(key.configuraion));
    hash = HashCombine(hash, std::hash<size_t>()(key.alignment));
    hash = HashCombine(hash, std::hash<size_t>()(key.kind));
    return hash;
  }
};

inline bool operator==(const cutlass::library::GemmArrayConfiguration& lhs,
                       const cutlass::library::GemmArrayConfiguration& rhs) {
  return lhs.batch_count == rhs.batch_count && lhs.problem_size == rhs.problem_size
         && lhs.lda == rhs.lda && lhs.ldb == rhs.ldb && lhs.ldc == rhs.ldc && lhs.ldd == rhs.ldd;
}

inline bool operator==(const GemmArrayOperationCacheKey& lhs,
                       const GemmArrayOperationCacheKey& rhs) {
  return lhs.functional_key == rhs.functional_key && lhs.configuraion == rhs.configuraion
         && lhs.alignment == rhs.alignment && lhs.kind == rhs.kind;
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUTLASS_GEMM_ARRAY_OPERATION_CACHE_KEY_H_

#endif  // WITH_CUTLASS
