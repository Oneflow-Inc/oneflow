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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_INFER_CACHE_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_INFER_CACHE_H_

#include "oneflow/core/operator/op_infer_cache.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"
#include "oneflow/core/kernel/kernel.pb.h"

namespace oneflow {

namespace user_op {

class KernelInferContext;

class OpKernelInferCache final {
 public:
  using KeyType = OpInferCacheKey;
  using ValueType = std::shared_ptr<const OpInferCacheValue>;
  using HashMap = std::unordered_map<HashEqTraitPtr<const KeyType>, ValueType>;
  using KeyStorage = std::list<std::unique_ptr<KeyType>>;
  static constexpr size_t kReleaseInIndependentThreadThreshold = 4096;

  OpKernelInferCache(const KernelConf& kernel_conf, const void* scope);
  ~OpKernelInferCache() = default;

  bool IsCacheHit() const;
  ValueType GetCacheValue() const;
  void UpdateCacheKey(KernelInferContext* ctx);
  void UpdateCacheValue(KernelInferContext* ctx);
  void Reset();

 private:
  KeyType cache_key_;
  HashMap cached_key2value_;
  KeyStorage key_storage_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_INFER_CACHE_H_
