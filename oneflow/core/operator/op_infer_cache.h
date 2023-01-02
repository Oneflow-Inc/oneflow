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
#ifndef ONEFLOW_CORE_OPERATOR_OP_INFER_CACHE_H_
#define ONEFLOW_CORE_OPERATOR_OP_INFER_CACHE_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/dtype_signature.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

struct OpInferCacheKey final {
  const void* scope;
  Symbol<OperatorConf> op_conf_sym;
  Symbol<DTypeSignature> dtype_signature_sym;
  std::vector<Symbol<Shape>> ibn_idx2shape_sym;
};

struct OpInferCacheValue final {
  std::vector<Symbol<Shape>> obn_idx2shape_sym;
};

inline bool operator==(const OpInferCacheKey& lhs, const OpInferCacheKey& rhs) {
  return lhs.scope == rhs.scope && lhs.op_conf_sym == rhs.op_conf_sym
         && lhs.dtype_signature_sym == rhs.dtype_signature_sym
         && lhs.ibn_idx2shape_sym == rhs.ibn_idx2shape_sym;
}

inline bool operator!=(const OpInferCacheKey& lhs, const OpInferCacheKey& rhs) {
  return !(lhs == rhs);
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::OpInferCacheKey> final {
  size_t operator()(const oneflow::OpInferCacheKey& op_infer_cache_key) const {
    using namespace oneflow;
    size_t ibn_idx2shape_sym_hash_value = 0;
    for (const auto& shape_sym : op_infer_cache_key.ibn_idx2shape_sym) {
      AddHash(&ibn_idx2shape_sym_hash_value, shape_sym);
    }
    return Hash(op_infer_cache_key.scope, op_infer_cache_key.op_conf_sym,
                ibn_idx2shape_sym_hash_value, op_infer_cache_key.dtype_signature_sym);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_OPERATOR_OP_INFER_CACHE_H_
