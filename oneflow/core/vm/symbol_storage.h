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
#ifndef ONEFLOW_CORE_VM_STORAGE_H_
#define ONEFLOW_CORE_VM_STORAGE_H_

#include <mutex>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class ParallelDesc;
class ParallelConf;

class OpNodeSignatureDesc;
class OpNodeSignature;

namespace vm {

template<typename T>
struct ConstructArgType4Symbol final {
  using type = T;
};

template<>
struct ConstructArgType4Symbol<OpNodeSignatureDesc> final {
  using type = OpNodeSignature;
};

template<typename T>
class SymbolStorage final {
 public:
  SymbolStorage(const SymbolStorage&) = delete;
  SymbolStorage(SymbolStorage&&) = delete;

  SymbolStorage() = default;
  ~SymbolStorage() = default;

  bool Has(int64_t symbol_id) const {
    std::unique_lock<std::mutex> lock(mutex_);
    return symbol_id2symbol_.find(symbol_id) != symbol_id2symbol_.end();
  }

  Maybe<const T&> MaybeGet(int64_t symbol_id) const { return *JUST(MaybeGetPtr(symbol_id)); }

  const T& Get(int64_t symbol_id) const { return *GetPtr(symbol_id); }

  Maybe<T> MaybeGetPtr(int64_t symbol_id) const {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto& iter = symbol_id2symbol_.find(symbol_id);
    CHECK_OR_RETURN(iter != symbol_id2symbol_.end()) << "symbol_id: " << symbol_id;
    return iter->second;
  }

  const std::shared_ptr<T>& GetPtr(int64_t symbol_id) const {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto& iter = symbol_id2symbol_.find(symbol_id);
    CHECK(iter != symbol_id2symbol_.end()) << "symbol_id: " << symbol_id;
    return iter->second;
  }

  void Add(int64_t symbol_id, const typename ConstructArgType4Symbol<T>::type& data) {
    CHECK_GT(symbol_id, 0);
    const auto& ptr = std::make_shared<T>(data);
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK(symbol_id2symbol_.emplace(symbol_id, ptr).second);
  }
  void Clear(int64_t symbol_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    symbol_id2symbol_.erase(symbol_id);
  }
  void ClearAll() {
    std::unique_lock<std::mutex> lock(mutex_);
    symbol_id2symbol_.clear();
  }

 private:
  mutable std::mutex mutex_;
  HashMap<int64_t, std::shared_ptr<T>> symbol_id2symbol_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STORAGE_H_
