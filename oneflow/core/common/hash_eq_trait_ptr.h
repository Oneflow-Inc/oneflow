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
