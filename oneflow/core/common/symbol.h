#ifndef ONEFLOW_CORE_COMMON_SYMBOL_H_
#define ONEFLOW_CORE_COMMON_SYMBOL_H_

namespace oneflow {

template<typename T>
class Symbol final {
 public:
  Symbol() : ptr_(nullptr) {}
  Symbol(const T& obj) : ptr_(FindOrInsertPtr(obj)) {}
  Symbol(const Symbol<T>&) = default;
  ~Symbol() = default;

  operator bool() const { return ptr_ != nullptr; }
  const T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }

  void reset() { ptr_ = nullptr; }
  void reset(const T& obj) { ptr_ = FindOrInsertPtr(obj); }

 private:
  static const T* FindOrInsertPtr(const T& obj);

  const T* ptr_;
};

template<typename T>
class HashEqTraitPtr final {
 public:
  HashEqTraitPtr(const HashEqTraitPtr<T>&) = default;
  HashEqTraitPtr(T* ptr, size_t hash_value) : ptr_(ptr), hash_value_(hash_value) {}
  ~HashEqTraitPtr() = default;

  T* ptr() const { return ptr_; }
  size_t hash_value() const { return hash_value_; }

  bool operator==(const HashEqTraitPtr<T>& rhs) const { return *ptr_ == *rhs.ptr_; }

 private:
  T* ptr_;
  size_t hash_value_;
};

template<typename T>
const T* Symbol<T>::FindOrInsertPtr(const T& obj) {
  static std::unordered_set<HashEqTraitPtr<const T>> cached_objs;
  size_t hash_value = std::hash<T>()(obj);
  {
    HashEqTraitPtr<const T> obj_ptr_wraper(&obj, hash_value);
    auto iter = cached_objs.find(obj_ptr_wraper);
    if (iter != cached_objs.end()) { return iter->ptr(); }
  }
  static std::mutex mutex;
  HashEqTraitPtr<const T> new_obj_ptr_wraper(new T(obj), hash_value);
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = cached_objs.find(new_obj_ptr_wraper);
  if (iter == cached_objs.end()) { iter = cached_objs.emplace(new_obj_ptr_wraper).first; }
  return iter->ptr();
}

}  // namespace oneflow

namespace std {

template<typename T>
struct hash<oneflow::HashEqTraitPtr<T>> final {
  size_t operator()(const oneflow::HashEqTraitPtr<T>& ptr) const { return ptr.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SYMBOL_H_
