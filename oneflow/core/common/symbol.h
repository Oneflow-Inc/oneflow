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

#include "oneflow/core/common/hash_eq_trait_ptr.h"

namespace oneflow {

template<typename T>
class Symbol final {
 public:
  Symbol() : ptr_(nullptr) {}
  Symbol(const T& obj) : ptr_(GetOrCreatePtr(obj)) {}
  Symbol(const Symbol<T>& rhs) = default;
  ~Symbol() = default;

  operator bool() const { return ptr_ != nullptr; }
  const T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }
  bool operator==(const Symbol<T>& rhs) const { return ptr_ == rhs.ptr_; }
  bool operator!=(const Symbol<T>& rhs) const { return !(*this == rhs); }
  size_t hash_value() const { return std::hash<const T*>()(ptr_); }

  void reset() { ptr_ = nullptr; }
  void reset(const T& obj) { ptr_ = GetOrCreatePtr(obj); }

 private:
  static const T* GetOrCreatePtr(const T& obj);

  const T* ptr_;
};

template<typename T>
const T* Symbol<T>::GetOrCreatePtr(const T& obj) {
  using HashSet = std::unordered_set<HashEqTraitPtr<const T>>;
  static HashSet cached_objs;
  static thread_local std::unordered_map<HashEqTraitPtr<const T>, const T*> obj2ptr;

  size_t hash_value = std::hash<T>()(obj);
  HashEqTraitPtr<const T> obj_ptr_wraper(&obj, hash_value);
  {
    auto iter = obj2ptr.find(obj_ptr_wraper);
    if (iter != obj2ptr.end()) { return iter->second; }
  }
  static std::mutex mutex;
  typename HashSet::iterator iter;
  {
    std::lock_guard<std::mutex> lock(mutex);
    iter = cached_objs.find(obj_ptr_wraper);
  }
  if (iter == cached_objs.end()) {
    HashEqTraitPtr<const T> new_obj_ptr_wraper(new T(obj), hash_value);
    std::lock_guard<std::mutex> lock(mutex);
    iter = cached_objs.emplace(new_obj_ptr_wraper).first;
  }
  obj2ptr[*iter] = iter->ptr();
  return iter->ptr();
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
