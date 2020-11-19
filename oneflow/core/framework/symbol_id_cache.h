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
#include <functional>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/scope.cfg.h"

namespace oneflow {

namespace symbol {

// Remove check util HasSymbolIdTraits after field ScopeProto.symbol_id removed.
template<typename T>
struct HasSymbolIdTraits;

template<typename T>
class IdCache final {
 public:
  IdCache(const IdCache&) = delete;
  IdCache(IdCache&&) = delete;

  IdCache() = default;
  ~IdCache() = default;

  Maybe<int64_t> FindOrCreate(const T& symbol_data, const std::function<Maybe<int64_t>()>& Create) {
    CHECK_OR_RETURN(!HasSymbolIdTraits<T>::HasSymbolId(symbol_data));
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

template<>
struct HasSymbolIdTraits<cfg::JobConfigProto> {
  static bool HasSymbolId(const cfg::JobConfigProto&) { return false; }
};

template<>
struct HasSymbolIdTraits<cfg::ParallelConf> {
  static bool HasSymbolId(const cfg::ParallelConf&) { return false; }
};

template<>
struct HasSymbolIdTraits<cfg::ScopeProto> {
  static bool HasSymbolId(const cfg::ScopeProto& scope) { return scope.has_symbol_id(); }
};

}  // namespace symbol

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SYMBOL_ID_CACHE_H_
