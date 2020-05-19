#ifndef ONEFLOW_CORE_COMMON_HASH_EQ_TRAIT_PTR_H_
#define ONEFLOW_CORE_COMMON_HASH_EQ_TRAIT_PTR_H_

namespace oneflow {

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

}  // namespace oneflow

namespace std {

template<typename T>
struct hash<oneflow::HashEqTraitPtr<T>> final {
  size_t operator()(const oneflow::HashEqTraitPtr<T>& ptr) const { return ptr.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_HASH_EQ_TRAIT_PTR_H_
