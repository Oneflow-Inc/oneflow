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
#ifndef ONEFLOW_CORE_INTRUSIVE_SHARED_PTR_H_
#define ONEFLOW_CORE_INTRUSIVE_SHARED_PTR_H_

#include "oneflow/core/intrusive/ref.h"

namespace oneflow {

namespace intrusive {

template<typename T>
class shared_ptr final {
 public:
  using value_type = T;
  shared_ptr() : ptr_(nullptr) {}
  shared_ptr(value_type* ptr) : ptr_(nullptr) { Reset(ptr); }
  shared_ptr(const shared_ptr& obj_ptr) {
    ptr_ = nullptr;
    Reset(obj_ptr.ptr_);
  }
  shared_ptr(shared_ptr&& obj_ptr) noexcept {
    ptr_ = obj_ptr.ptr_;
    obj_ptr.ptr_ = nullptr;
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator shared_ptr<const T>() const { return shared_ptr<const T>(ptr_); }
  ~shared_ptr() { Clear(); }

  template<typename... Args>
  static shared_ptr make_shared(Args&&... args) {
    shared_ptr ret;
    Ref::NewAndInitRef(&ret.ptr_);
    ret.Mutable()->__Init__(std::forward<Args>(args)...);
    return ret;
  }

  explicit operator bool() const { return ptr_ != nullptr; }
  value_type* get() const { return ptr_; }
  const value_type& Get() const { return *ptr_; }
  value_type* operator->() const { return ptr_; }
  value_type& operator*() const { return *ptr_; }
  bool operator==(const shared_ptr& rhs) const { return this->ptr_ == rhs.ptr_; }

  value_type* Mutable() { return ptr_; }

  void Reset() { Reset(nullptr); }

  void Reset(value_type* ptr) {
    Clear();
    if (ptr == nullptr) { return; }
    ptr_ = ptr;
    Ref::IncreaseRef<value_type>(ptr_);
  }

  shared_ptr& operator=(const shared_ptr& rhs) {
    Reset(rhs.ptr_);
    return *this;
  }

  shared_ptr& operator=(shared_ptr&& rhs) noexcept {
    ptr_ = rhs.ptr_;
    rhs.ptr_ = nullptr;
    return *this;
  }

  static shared_ptr __UnsafeMove__(value_type* ptr) {
    shared_ptr ret;
    ret.ptr_ = ptr;
    return ret;
  }
  void __UnsafeMoveTo__(value_type** ptr) {
    *ptr = ptr_;
    ptr_ = nullptr;
  }

 private:
  void Clear() {
    if (ptr_ == nullptr) { return; }
    Ref::DecreaseRef<value_type>(ptr_);
    ptr_ = nullptr;
  }
  mutable value_type* ptr_;
};

template<typename T, typename... Args>
shared_ptr<T> make_shared(Args&&... args) {
  return shared_ptr<T>::make_shared(std::forward<Args>(args)...);
}

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_SHARED_PTR_H_
