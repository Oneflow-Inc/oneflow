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
#ifndef ONEFLOW_CORE_FRAMEWORK_SYMBOL_ID_CACHE_H_
#define ONEFLOW_CORE_FRAMEWORK_SYMBOL_ID_CACHE_H_

#include <mutex>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/scope.cfg.h"

namespace oneflow {

template<typename T, bool (&HasSymbolId)(const T& symbol_data)>
class SymbolIdCache final {
 public:
  SymbolIdCache(const SymbolIdCache&) = delete;
  SymbolIdCache(SymbolIdCache&&) = delete;

  SymbolIdCache() = default;
  ~SymbolIdCache() = default;

  template<typename CreateT>
  Maybe<int64_t> FindOrCreate(const T& symbol_data, const CreateT& Create) {
    CHECK_OR_RETURN(!HasSymbolId(symbol_data));
    {
      std::unique_lock<std::mutex> lock(mutex_);
      const auto& iter = symbol_data2id_.find(symbol_data);
      if (iter != symbol_data2id_.end()) { return iter->second; }
    }
    int64_t symbol_id = JUST(Create());
    {
      std::unique_lock<std::mutex> lock(mutex_);
      symbol_data2id_[symbol_data] = symbol_id;
    }
    return symbol_id;
  }

 private:
  mutable std::mutex mutex_;
  std::map<T, int64_t> symbol_data2id_;
};

inline bool JobConfigProtoHasSymbolId(const cfg::JobConfigProto&) { return false; }
using JobConfSymbolIdCache = SymbolIdCache<cfg::JobConfigProto, JobConfigProtoHasSymbolId>;

inline bool ParallelConfHasSymbolId(const cfg::ParallelConf&) { return false; }
using ParallelConfSymbolIdCache = SymbolIdCache<cfg::ParallelConf, ParallelConfHasSymbolId>;

inline bool ScopeProtoHasSymbolId(const cfg::ScopeProto& scope) { return scope.has_symbol_id(); }
using ScopeSymbolIdCache = SymbolIdCache<cfg::ScopeProto, ScopeProtoHasSymbolId>;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SYMBOL_ID_CACHE_H_
