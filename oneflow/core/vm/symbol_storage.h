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

class StringSymbol;

class OperatorConfSymbol;
class OperatorConf;

class ParallelDesc;
class ParallelConf;

class JobDesc;
class JobConfigProto;

class OpNodeSignatureDesc;
class OpNodeSignature;

class Scope;
class ScopeProto;

namespace symbol {

template<typename T>
struct ConstructArgType4Symbol final {
  using type = T;
};

template<>
struct ConstructArgType4Symbol<OpNodeSignatureDesc> final {
  using type = OpNodeSignature;
};

template<>
struct ConstructArgType4Symbol<StringSymbol> final {
  using type = std::string;
};

template<>
struct ConstructArgType4Symbol<OperatorConfSymbol> final {
  using type = OperatorConf;
};

template<>
struct ConstructArgType4Symbol<ParallelDesc> final {
  using type = ParallelConf;
};

template<>
struct ConstructArgType4Symbol<JobDesc> final {
  using type = JobConfigProto;
};

template<>
struct ConstructArgType4Symbol<Scope> final {
  using type = ScopeProto;
};

namespace detail {

template<typename T>
Maybe<T> NewSymbol(int64_t symbol_id, const typename ConstructArgType4Symbol<T>::type& data) {
  return std::make_shared<T>(data);
}

template<>
Maybe<StringSymbol> NewSymbol<StringSymbol>(
    int64_t symbol_id, const typename ConstructArgType4Symbol<StringSymbol>::type& data);

template<>
Maybe<OperatorConfSymbol> NewSymbol<OperatorConfSymbol>(
    int64_t symbol_id, const typename ConstructArgType4Symbol<OperatorConfSymbol>::type& data);

template<>
Maybe<ParallelDesc> NewSymbol<ParallelDesc>(
    int64_t symbol_id, const typename ConstructArgType4Symbol<ParallelDesc>::type& data);

template<>
Maybe<JobDesc> NewSymbol<JobDesc>(int64_t symbol_id,
                                  const typename ConstructArgType4Symbol<JobDesc>::type& data);

template<>
Maybe<Scope> NewSymbol<Scope>(int64_t symbol_id,
                              const typename ConstructArgType4Symbol<Scope>::type& data);

template<>
Maybe<OpNodeSignatureDesc> NewSymbol<OpNodeSignatureDesc>(
    int64_t symbol_id, const typename ConstructArgType4Symbol<OpNodeSignatureDesc>::type& data);

}  // namespace detail

template<typename T>
class Storage final {
 public:
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;

  Storage() = default;
  ~Storage() = default;

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

  Maybe<void> Add(int64_t symbol_id, const typename ConstructArgType4Symbol<T>::type& data) {
    CHECK_GT_OR_RETURN(symbol_id, 0);
    const auto& ptr = JUST(detail::NewSymbol<T>(symbol_id, data));
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK_OR_RETURN(symbol_id2symbol_.emplace(symbol_id, ptr).second);
    return Maybe<void>::Ok();
  }

  Maybe<void> TryAdd(int64_t symbol_id, const typename ConstructArgType4Symbol<T>::type& data) {
    CHECK_GT_OR_RETURN(symbol_id, 0);
    const auto& ptr = JUST(detail::NewSymbol<T>(symbol_id, data));
    std::unique_lock<std::mutex> lock(mutex_);
    const auto& iter = symbol_id2symbol_.find(symbol_id);
    if (iter != symbol_id2symbol_.end()) { return Maybe<void>::Ok(); }
    CHECK_OR_RETURN(symbol_id2symbol_.emplace(symbol_id, ptr).second);
    return Maybe<void>::Ok();
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

}  // namespace symbol
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STORAGE_H_
