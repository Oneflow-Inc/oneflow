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

#ifndef ONEFLOW_USER_KERNELS_CUTLASS_CONV2D_OPERATION_CACHE_KEY_H_
#define ONEFLOW_USER_KERNELS_CUTLASS_CONV2D_OPERATION_CACHE_KEY_H_

#include "oneflow/core/framework/framework.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>

#ifdef WITH_CUTLASS_EXTENSION
#include <cutlass/library/cutlass_extension_library.h>
#endif  // WITH_CUTLASS_EXTENSION

namespace oneflow {

struct Conv2dOperationCacheKey {
  cutlass::library::ConvFunctionalKey functional_key;
  cutlass::library::Conv2dConfiguration configuraion;
  size_t alignment;
  Conv2dOperationCacheKey(cutlass::library::ConvFunctionalKey functional_key,
                          cutlass::library::Conv2dConfiguration configuraion,
                          cutlass::library::ConvArguments arguments)
      : functional_key(functional_key), configuraion(configuraion) {
    const auto IsStrideAligned = [&](const std::vector<int64_t>& stride, size_t n) {
      return std::all_of(stride.cbegin(), stride.cend(),
                         [&](const int64_t& s) { return s % n == 0; });
    };
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.A) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.B) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.C) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.D) % kCudaAlignSize, 0);
    const auto IsAligned = [&](size_t n) {
      return IsStrideAligned(configuraion.stride_a, n) && IsStrideAligned(configuraion.stride_b, n)
             && IsStrideAligned(configuraion.stride_c, n);
    };
    alignment = 128 / cutlass::library::sizeof_bits(functional_key.element_A);
    for (; alignment > 1; alignment = alignment >> 1) {
      if (IsAligned(alignment)) { break; }
    }
  }

#ifdef WITH_CUTLASS_EXTENSION
  Conv2dOperationCacheKey(cutlass::library::ConvFunctionalKey functional_key,
                          const cutlass::library::Conv2dScaleBiasFusionConfiguration& config,
                          const cutlass::library::ConvScaleBiasFusionArguments& arguments)
      : functional_key(functional_key) {
    configuraion.problem_size = config.problem_size;
    configuraion.split_k_mode = config.split_k_mode;
    configuraion.stride_a = config.stride_a;
    configuraion.stride_b = config.stride_b;
    configuraion.stride_c = {0, 0, 0};
    const auto IsStrideAligned = [&](const std::vector<int64_t>& stride, size_t n) {
      return std::all_of(stride.cbegin(), stride.cend(),
                         [&](const int64_t& s) { return s % n == 0; });
    };
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.A) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.B) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.P) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.Scale) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.Bias) % kCudaAlignSize, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.D) % kCudaAlignSize, 0);
    const auto IsAligned = [&](size_t n) {
      return IsStrideAligned(configuraion.stride_a, n) && IsStrideAligned(configuraion.stride_b, n)
             && IsStrideAligned(configuraion.stride_c, n);
    };
    alignment = 128 / cutlass::library::sizeof_bits(functional_key.element_A);
    for (; alignment > 1; alignment = alignment >> 1) {
      if (IsAligned(alignment)) { break; }
    }
  }
#endif  // WITH_CUTLASS_EXTENSION
};

struct Conv2dProblemSizeHasher {
  size_t operator()(const cutlass::conv::Conv2dProblemSize& problem_size) const {
    size_t hash = 0;
    hash = HashCombine(hash, std::hash<int>()(problem_size.N));
    hash = HashCombine(hash, std::hash<int>()(problem_size.H));
    hash = HashCombine(hash, std::hash<int>()(problem_size.W));
    hash = HashCombine(hash, std::hash<int>()(problem_size.C));
    hash = HashCombine(hash, std::hash<int>()(problem_size.P));
    hash = HashCombine(hash, std::hash<int>()(problem_size.Q));
    hash = HashCombine(hash, std::hash<int>()(problem_size.K));
    hash = HashCombine(hash, std::hash<int>()(problem_size.R));
    hash = HashCombine(hash, std::hash<int>()(problem_size.S));
    hash = HashCombine(hash, std::hash<int>()(problem_size.pad_h));
    hash = HashCombine(hash, std::hash<int>()(problem_size.pad_w));
    hash = HashCombine(hash, std::hash<int>()(problem_size.stride_h));
    hash = HashCombine(hash, std::hash<int>()(problem_size.stride_w));
    hash = HashCombine(hash, std::hash<int>()(problem_size.dilation_h));
    hash = HashCombine(hash, std::hash<int>()(problem_size.dilation_w));
    hash = HashCombine(hash, std::hash<int>()(static_cast<int>(problem_size.mode)));
    hash = HashCombine(hash, std::hash<int>()(problem_size.split_k_slices));
    hash = HashCombine(hash, std::hash<int>()(problem_size.groups));
    return hash;
  }
};

struct Conv2dConfigurationHasher {
  size_t operator()(const cutlass::library::Conv2dConfiguration& configuraion) const {
    size_t hash = std::hash<int>()(static_cast<int>(configuraion.split_k_mode));
    hash = HashCombine(hash, Conv2dProblemSizeHasher()(configuraion.problem_size));
    for (const int64_t v : configuraion.stride_a) {
      hash = HashCombine(hash, std::hash<int64_t>()(v));
    }
    for (const int64_t v : configuraion.stride_b) {
      hash = HashCombine(hash, std::hash<int64_t>()(v));
    }
    for (const int64_t v : configuraion.stride_c) {
      hash = HashCombine(hash, std::hash<int64_t>()(v));
    }
    return hash;
  }
};

struct Conv2dOperationCacheKeyHasher {
  size_t operator()(const Conv2dOperationCacheKey& key) const {
    size_t hash = cutlass::library::ConvFunctionalKeyHasher()(key.functional_key);
    hash = HashCombine(hash, Conv2dConfigurationHasher()(key.configuraion));
    hash = HashCombine(hash, std::hash<size_t>()(key.alignment));
    return hash;
  }
};

inline bool operator==(const cutlass::library::Conv2dConfiguration& lhs,
                       const cutlass::library::Conv2dConfiguration& rhs) {
  return lhs.split_k_mode == rhs.split_k_mode && lhs.problem_size == rhs.problem_size
         && lhs.stride_a == rhs.stride_a && lhs.stride_b == rhs.stride_b
         && lhs.stride_c == rhs.stride_c;
}

inline bool operator==(const Conv2dOperationCacheKey& lhs, const Conv2dOperationCacheKey& rhs) {
  return lhs.functional_key == rhs.functional_key && lhs.configuraion == rhs.configuraion
         && lhs.alignment == rhs.alignment;
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUTLASS_CONV2D_OPERATION_CACHE_KEY_H_

#endif  // WITH_CUTLASS
