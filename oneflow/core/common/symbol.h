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
#include <glog/logging.h>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"

namespace oneflow {

template<typename T>
class Symbol final {
 public:
  Symbol() : ptr_(nullptr) {}
  Symbol(const T& obj) : ptr_(GetOrCreatePtr(obj)) {}
  Symbol(const Symbol& rhs) = default;
  Symbol(Symbol&& rhs) = default;
  ~Symbol() = default;

  operator bool() const { return ptr_ != nullptr; }
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

  std::shared_ptr<const T> shared_from_symbol() const;

 private:
  static const T* GetOrCreatePtr(const T& obj);

  const T* ptr_;
};

template<typename T>
struct IsScalarType<Symbol<T>> final {
  static const bool value = true;
};

namespace sym {
template<typename T>
using SymbolTable = std::unordered_map<HashEqTraitPtr<const T>, std::shared_ptr<const T>>;

template<typename T>
SymbolTable<T>* GlobalSymbolTable() {
  static SymbolTable<T> symbol_table;
  return &symbol_table;
}

template<typename T>
std::mutex* GlobalSymbolTableMutex() {
  static std::mutex mutex;
  return &mutex;
}

template<typename T>
SymbolTable<T>* ThreadLocalSymbolTable() {
  static thread_local SymbolTable<T> thread_local_symbol_table;
  return &thread_local_symbol_table;
}

template<typename T,
         typename SymbolTable<T>::iterator (*GetIter4ObjectAndHashValue)(const T&, size_t)>
std::shared_ptr<const T> LocalThreadGetOr(const T& obj) {
  auto* thread_local_symbol_table = ThreadLocalSymbolTable<T>();
  size_t hash_value = std::hash<T>()(obj);
  HashEqTraitPtr<const T> obj_ptr_wraper(&obj, hash_value);
  const auto& local_iter = thread_local_symbol_table->find(obj_ptr_wraper);
  if (local_iter != thread_local_symbol_table->end()) { return local_iter->second; }
  const auto& iter = GetIter4ObjectAndHashValue(obj, hash_value);
  (*thread_local_symbol_table)[iter->first] = iter->second;
  return iter->second;
}

template<typename T>
typename SymbolTable<T>::iterator FindGlobalSymbol(const T& obj, size_t hash_value) {
  HashEqTraitPtr<const T> new_obj_ptr_wraper(&obj, hash_value);
  auto* symbol_table = GlobalSymbolTable<T>();
  std::unique_lock<std::mutex> lock(*GlobalSymbolTableMutex<T>());
  const auto& iter = symbol_table->find(new_obj_ptr_wraper);
  CHECK(iter != symbol_table->end());
  return iter;
}

template<typename T>
std::shared_ptr<const T> SharedFromObject(const T& obj) {
  return LocalThreadGetOr<T, FindGlobalSymbol<T>>(obj);
}

template<typename T>
typename SymbolTable<T>::iterator CreateGlobalSymbol(const T& obj, size_t hash_value) {
  std::shared_ptr<const T> ptr(new T(obj));
  HashEqTraitPtr<const T> new_obj_ptr_wraper(ptr.get(), hash_value);
  std::unique_lock<std::mutex> lock(*GlobalSymbolTableMutex<T>());
  return GlobalSymbolTable<T>()->emplace(new_obj_ptr_wraper, ptr).first;
}

template<typename T>
std::shared_ptr<const T> GetOrCreatePtr(const T& obj) {
  return LocalThreadGetOr<T, CreateGlobalSymbol<T>>(obj);
}

}  // namespace sym

template<typename T>
std::shared_ptr<const T> Symbol<T>::shared_from_symbol() const {
  if (this->ptr_ == nullptr) { return std::shared_ptr<const T>(); }
  return sym::SharedFromObject(*this->ptr_);
}

template<typename T>
const T* Symbol<T>::GetOrCreatePtr(const T& obj) {
  return sym::GetOrCreatePtr(obj).get();
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
