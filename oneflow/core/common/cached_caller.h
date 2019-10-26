#ifndef ONEFLOW_CORE_COMMON_CACHED_CALLER_H_
#define ONEFLOW_CORE_COMMON_CACHED_CALLER_H_

#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"

namespace oneflow {

template<typename Ret, typename Arg>
class CachedCaller final {
 public:
  CachedCaller(size_t max_size) : max_size_(max_size) {}
  CachedCaller(const CachedCaller& rhs) : max_size_(rhs.max_size_), cache_(rhs.cache_) {}
  ~CachedCaller() { Clear(); }

  const Ret& operator()(const std::function<Ret(const Arg&)>& f, const Arg& arg) {
    if (cache_.size() >= max_size_) { Clear(); }
    size_t hash_value = std::hash<Arg>()(arg);
    {
      HashEqTraitPtr<const Arg> ptr_wraper(&arg, hash_value);
      std::lock_guard<std::mutex> lock(mutex_);
      auto iter = cache_.find(ptr_wraper);
      if (iter != cache_.end()) { return iter->second; }
    }
    {
      HashEqTraitPtr<const Arg> ptr_wraper(new Arg(arg), hash_value);
      std::lock_guard<std::mutex> lock(mutex_);
      return cache_.emplace(ptr_wraper, f(arg)).first->second;
    }
  }

 private:
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pair : cache_) { delete pair.first.ptr(); }
    cache_.clear();
  }

  const size_t max_size_;
  std::unordered_map<HashEqTraitPtr<const Arg>, Ret> cache_;
  std::mutex mutex_;
};

template<
    typename F, typename Ret = typename function_traits<F>::return_type,
    typename RawArg = typename std::tuple_element<0, typename function_traits<F>::args_type>::type,
    typename Arg = typename std::remove_const<typename std::remove_reference<RawArg>::type>::type>
std::function<Ret(const Arg&)> WithResultCached(F f) {
  auto cached = std::make_shared<std::unordered_map<Arg, Ret>>();
  return [cached, f](const Arg& arg) -> Ret {
    const auto& iter = cached->find(arg);
    if (iter != cached->end()) { return iter->second; }
    return cached->emplace(arg, f(arg)).first->second;
  };
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CACHED_CALLER_H_
