#ifndef ONEFLOW_CORE_COMMON_SYMBOL_H_
#define ONEFLOW_CORE_COMMON_SYMBOL_H_

#include "oneflow/core/common/hash_eq_trait_ptr.h"

namespace oneflow {

template<typename T>
class Symbol final {
 public:
  Symbol() : ptr_(nullptr) {}
  Symbol(const T& obj) : ptr_(FindOrInsertPtr(obj)) {}
  Symbol(const Symbol<T>& rhs) = default;
  ~Symbol() = default;

  operator bool() const { return ptr_ != nullptr; }
  const T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }
  bool operator==(const Symbol<T>& rhs) { return ptr_ == rhs.ptr_; }
  size_t hash_value() const { return std::hash<const T*>()(ptr_); }

  void reset() { ptr_ = nullptr; }
  void reset(const T& obj) { ptr_ = FindOrInsertPtr(obj); }

 private:
  static const T* FindOrInsertPtr(const T& obj);

  const T* ptr_;
};

template<typename T>
const T* Symbol<T>::FindOrInsertPtr(const T& obj) {
  static std::unordered_set<HashEqTraitPtr<const T>> cached_objs;
  static std::mutex mutex;
  size_t hash_value = std::hash<T>()(obj);
  {
    HashEqTraitPtr<const T> obj_ptr_wraper(&obj, hash_value);
    std::lock_guard<std::mutex> lock(mutex);
    auto iter = cached_objs.find(obj_ptr_wraper);
    if (iter != cached_objs.end()) { return iter->ptr(); }
  }
  {
    HashEqTraitPtr<const T> new_obj_ptr_wraper(new T(obj), hash_value);
    std::lock_guard<std::mutex> lock(mutex);
    return cached_objs.emplace(new_obj_ptr_wraper).first->ptr();
  }
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
