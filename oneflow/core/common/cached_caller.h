#ifndef ONEFLOW_CORE_COMMON_CACHED_CALLER_H_
#define ONEFLOW_CORE_COMMON_CACHED_CALLER_H_

#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"

namespace oneflow {

template<typename F>
class CachedCaller final {
 public:
  CachedCaller(size_t max_size, F f) : max_size_(max_size), f_(f) {}
  CachedCaller(const CachedCaller& rhs)
      : max_size_(rhs.max_size_), f_(rhs.f_), cache_(rhs.cache_) {}
  ~CachedCaller() { Clear(); }

  static_assert(std::tuple_size<typename function_traits<F>::args_type>::value == 1,
                "only sole argument functions supported");
  using RawRet = typename function_traits<F>::return_type;
  using RawArg = typename std::tuple_element<0, typename function_traits<F>::args_type>::type;
  using R = typename std::remove_const<typename std::remove_reference<RawRet>::type>::type;
  using Arg = typename std::remove_const<typename std::remove_reference<RawArg>::type>::type;

  const R& operator()(const Arg& arg) {
    if (cache_.size() >= max_size_) { Clear(); }
    size_t hash_value = std::hash<Arg>()(arg);
    {
      HashEqTraitPtr<const Arg> ptr_wraper(&arg, hash_value);
      std::unique_lock<std::mutex> lock(mutex_);
      auto iter = cache_.find(ptr_wraper);
      if (iter != cache_.end()) { return iter->second; }
    }
    {
      HashEqTraitPtr<const Arg> ptr_wraper(new Arg(arg), hash_value);
      std::unique_lock<std::mutex> lock(mutex_);
      auto iter = cache_.find(ptr_wraper);
      if (iter == cache_.end()) { iter = cache_.emplace(ptr_wraper, f_(arg)).first; }
      return iter->second;
    }
  }

 private:
  void Clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto& pair : cache_) { delete pair.first.ptr(); }
    cache_.clear();
  }

  const size_t max_size_;
  F f_;
  std::unordered_map<HashEqTraitPtr<const Arg>, R> cache_;
  std::mutex mutex_;
};

template<typename F>
CachedCaller<F> MakeCachedCaller(size_t max_size, F f) {
  return CachedCaller<F>(max_size, f);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CACHED_CALLER_H_
