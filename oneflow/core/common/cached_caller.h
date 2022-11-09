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
#ifndef ONEFLOW_CORE_COMMON_CACHED_CALLER_H_
#define ONEFLOW_CORE_COMMON_CACHED_CALLER_H_

#include <list>
#include <tuple>
#include <thread>
#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/tuple_hash.h"

// gcc 11 falsely reports error:
// ‘void operator delete(void*, std::size_t)’ called on unallocated object ‘cache’
// However, `DeleteAndClear` is only called after `cache` is allocated in
// if (cache == nullptr) block.
// The reason not to use #pragma GCC diagnostic push/pop is that gcc reports
// the error on the caller of `ThreadLocalCachedCall`.
// TODO: replace ThreadLocalCachedCall with ThreadLocalCached decorator?
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif

namespace oneflow {

template<typename T>
void DeleteAndClear(T** ptr, size_t obj_cnt) {
  static const size_t kThreshold = 4096;
  if (obj_cnt <= kThreshold) {
    delete ptr;
  } else {
    std::thread([](T* ptr) { delete ptr; }, *ptr);
  }
  *ptr = nullptr;
}

bool IsThreadLocalCacheEnabled();

template<
    typename F, typename Ret = typename function_traits<F>::return_type,
    typename RawArg = typename std::tuple_element<0, typename function_traits<F>::args_type>::type,
    typename Arg = typename std::remove_const<typename std::remove_reference<RawArg>::type>::type>
Ret ThreadLocalCachedCall(size_t max_size, F f, const Arg& arg) {
  if (IsThreadLocalCacheEnabled() == false) { return f(arg); }
  using HashMap = std::unordered_map<HashEqTraitPtr<const Arg>, Ret>;
  using KeyStorage = std::list<std::unique_ptr<Arg>>;
  static thread_local HashMap* cache = nullptr;
  static thread_local KeyStorage* key_storage = nullptr;
  if (cache != nullptr && cache->size() >= max_size) {
    DeleteAndClear(&cache, cache->size());
    DeleteAndClear(&key_storage, cache->size());
  }
  if (cache == nullptr) {
    cache = new HashMap();
    key_storage = new KeyStorage();
  }
  size_t hash_value = std::hash<Arg>()(arg);
  {
    HashEqTraitPtr<const Arg> ptr_wrapper(&arg, hash_value);
    const auto& iter = cache->find(ptr_wrapper);
    if (iter != cache->end()) { return iter->second; }
  }
  Arg* new_arg = new Arg(arg);
  key_storage->emplace_back(new_arg);
  HashEqTraitPtr<const Arg> ptr_wrapper(new_arg, hash_value);
  return cache->emplace(ptr_wrapper, f(*new_arg)).first->second;
}

template<
    typename F, typename Ret = typename function_traits<F>::return_type,
    typename RawArg = typename std::tuple_element<0, typename function_traits<F>::args_type>::type,
    typename Arg = typename std::remove_const<typename std::remove_reference<RawArg>::type>::type>
std::function<Ret(const Arg&)> WithResultCached(F f) {
  auto cache = std::make_shared<std::unordered_map<Arg, Ret>>();
  return [cache, f](const Arg& arg) -> Ret {
    const auto& iter = cache->find(arg);
    if (iter != cache->end()) { return iter->second; }
    return cache->emplace(arg, f(arg)).first->second;
  };
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CACHED_CALLER_H_
