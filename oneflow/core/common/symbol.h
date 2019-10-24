#ifndef ONEFLOW_CORE_COMMON_SYMBOL_H_
#define ONEFLOW_CORE_COMMON_SYMBOL_H_

namespace oneflow {

template<typename T>
class Symbol final {
 public:
  Symbol() : ptr_(nullptr) {}
  Symbol(const Symbol<T>&) = default;

  bool HasValue() const { return ptr_ != nullptr; }
  const T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }

  void Clear() { ptr_ = nullptr; }

  static Symbol<T> Of(const T& obj);

 private:
  Symbol(const T* ptr) : ptr_(ptr) {}
  const T* ptr_;
};

template<typename T>
class HashEqTraitPtr final {
 public:
  HashEqTraitPtr(const HashEqTraitPtr<T>&) = default;
  HashEqTraitPtr() : ptr_(nullptr) {}
  HashEqTraitPtr(T* ptr) : ptr_(ptr) {}

  T* ptr() const { return ptr_; }

  bool operator==(const HashEqTraitPtr<T>& rhs) const { return *ptr_ == *rhs.ptr_; }

 private:
  T* ptr_;
};

template<typename T>
Symbol<T> Symbol<T>::Of(const T& obj) {
  static std::unordered_set<HashEqTraitPtr<const T>> cached_objs;
  static std::mutex mutex;

  HashEqTraitPtr<const T> obj_ptr_wraper(new T(obj));
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = cached_objs.find(obj_ptr_wraper);
  if (iter == cached_objs.end()) { iter = cached_objs.emplace(obj_ptr_wraper).first; }
  return iter->ptr();
}

}  // namespace oneflow

namespace std {

template<typename T>
struct hash<oneflow::HashEqTraitPtr<T>> final {
  using VT = typename std::remove_const<T>::type;
  size_t operator()(const oneflow::HashEqTraitPtr<T>& ptr) const { return hash<VT>()(*ptr.ptr()); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SYMBOL_H_
