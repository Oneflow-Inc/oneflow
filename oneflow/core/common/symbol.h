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
#ifndef ONEFLOW_CORE_COMMON_SYMBOL_H_
#define ONEFLOW_CORE_COMMON_SYMBOL_H_

#include <mutex>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <glog/logging.h>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"

namespace oneflow {

template<typename T>
struct SymbolUtil;

template<typename T>
class Symbol final {
 public:
  Symbol() : ptr_(nullptr) {}
  Symbol(const T& obj) : ptr_(GetOrCreatePtr(obj)) {}
  Symbol(const Symbol& rhs) = default;
  Symbol(Symbol&& rhs) = default;
  ~Symbol() = default;

  explicit operator bool() const { return ptr_ != nullptr; }
  const T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }
  bool operator==(const Symbol<T>& rhs) const { return ptr_ == rhs.ptr_; }
  bool operator!=(const Symbol<T>& rhs) const { return !(*this == rhs); }
  size_t hash_value() const { return std::hash<const T*>()(ptr_); }

  Symbol& operator=(const Symbol& other) {
    ptr_ = other.ptr_;
    return *this;
  }
  void reset() { ptr_ = nullptr; }
  void reset(const T& obj) { ptr_ = GetOrCreatePtr(obj); }

  const std::shared_ptr<const T>& shared_from_symbol() const;

 private:
  template<typename SymbolT>
  friend struct SymbolUtil;
  static const T* GetOrCreatePtr(const T& obj);

  const T* ptr_;
};

template<typename T>
struct IsScalarType<Symbol<T>> final {
  static const bool value = true;
};

template<typename T>
struct SymbolUtil final {
  using SymbolMap = std::unordered_map<HashEqTraitPtr<const T>, std::shared_ptr<const T>>;

  static SymbolMap* GlobalSymbolMap() {
    static SymbolMap symbol_map;
    return &symbol_map;
  }

  static std::mutex* GlobalSymbolMapMutex() {
    static std::mutex mutex;
    return &mutex;
  }

  static SymbolMap* ThreadLocalSymbolMap() {
    static thread_local SymbolMap thread_local_symbol_map;
    return &thread_local_symbol_map;
  }

  static std::unordered_set<const T*>* ThreadLocalSymbolPtrSet() {
    static thread_local std::unordered_set<const T*> thread_local_symbol_ptr_set;
    return &thread_local_symbol_ptr_set;
  }

  template<typename SymbolMap::iterator (*GetIter4ObjectAndHashValue)(const T&, size_t)>
  static const std::shared_ptr<const T>& LocalThreadGetOr(const T& obj) {
    auto* thread_local_symbol_map = ThreadLocalSymbolMap();
    size_t hash_value = std::hash<T>()(obj);
    HashEqTraitPtr<const T> obj_ptr_wraper(&obj, hash_value);
    const auto& local_iter = thread_local_symbol_map->find(obj_ptr_wraper);
    if (local_iter != thread_local_symbol_map->end()) { return local_iter->second; }
    const auto& iter = GetIter4ObjectAndHashValue(obj, hash_value);
    (*thread_local_symbol_map)[iter->first] = iter->second;
    CHECK(ThreadLocalSymbolPtrSet()->emplace(iter->second.get()).second);
    return iter->second;
  }

  static typename SymbolMap::iterator FindGlobalSymbol(const T& obj, size_t hash_value) {
    HashEqTraitPtr<const T> new_obj_ptr_wraper(&obj, hash_value);
    auto* symbol_map = GlobalSymbolMap();
    std::unique_lock<std::mutex> lock(*GlobalSymbolMapMutex());
    const auto& iter = symbol_map->find(new_obj_ptr_wraper);
    CHECK(iter != symbol_map->end());
    return iter;
  }

  static const std::shared_ptr<const T>& SharedFromObject(const T& obj) {
    return LocalThreadGetOr<FindGlobalSymbol>(obj);
  }

  static typename SymbolMap::iterator CreateGlobalSymbol(const T& obj, size_t hash_value) {
    std::shared_ptr<const T> ptr(new T(obj));
    HashEqTraitPtr<const T> new_obj_ptr_wraper(ptr.get(), hash_value);
    std::unique_lock<std::mutex> lock(*GlobalSymbolMapMutex());
    return GlobalSymbolMap()->emplace(new_obj_ptr_wraper, ptr).first;
  }

  static const std::shared_ptr<const T>& GetOrCreatePtr(const T& obj) {
    return LocalThreadGetOr<CreateGlobalSymbol>(obj);
  }
};

template<typename T>
const std::shared_ptr<const T>& Symbol<T>::shared_from_symbol() const {
  if (this->ptr_ == nullptr) {
    static auto* none = new std::shared_ptr<const T>();
    return *none;
  }
  return SymbolUtil<T>::SharedFromObject(*this->ptr_);
}

template<typename T>
const T* Symbol<T>::GetOrCreatePtr(const T& obj) {
  return SymbolUtil<T>::GetOrCreatePtr(obj).get();
}

template<typename T>
Symbol<T> SymbolOf(const T& obj) {
  return Symbol<T>(obj);
}

}  // namespace oneflow

namespace std {

template<typename T>
struct hash<oneflow::Symbol<T>> final {
  size_t operator()(const oneflow::Symbol<T>& symbol) const { return symbol.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SYMBOL_H_
