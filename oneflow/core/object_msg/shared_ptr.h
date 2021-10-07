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
#ifndef ONEFLOW_CORE_OBJECT_MSG_SHARED_PTR_H_
#define ONEFLOW_CORE_OBJECT_MSG_SHARED_PTR_H_

#include "oneflow/core/object_msg/object_msg_core.h"

namespace oneflow {

namespace intrusive {

template<typename T>
class SharedPtr final {
 public:
  static_assert(T::__has_intrusive_ref__, "T is not a intrusive-referenced class");
  using value_type = T;
  SharedPtr() : ptr_(nullptr) {}
  SharedPtr(value_type* ptr) : ptr_(nullptr) { Reset(ptr); }
  SharedPtr(const SharedPtr& obj_ptr) {
    ptr_ = nullptr;
    Reset(obj_ptr.ptr_);
  }
  SharedPtr(SharedPtr&& obj_ptr) {
    ptr_ = obj_ptr.ptr_;
    obj_ptr.ptr_ = nullptr;
  }
  ~SharedPtr() { Clear(); }

  template<typename... Args>
  static SharedPtr MakeShared(Args&&... args) {
    SharedPtr ret;
    PtrUtil::NewAndInitRef(&ret.ptr_);
    ret.Mutable()->__Init__(std::forward<Args>(args)...);
    return ret;
  }

  operator bool() const { return ptr_ != nullptr; }
  const value_type& Get() const { return *ptr_; }
  const value_type* operator->() const { return ptr_; }
  const value_type& operator*() const { return *ptr_; }
  bool operator==(const SharedPtr& rhs) const { return this->ptr_ == rhs.ptr_; }

  value_type* Mutable() { return ptr_; }
  value_type* operator->() { return ptr_; }
  value_type& operator*() { return *ptr_; }

  void Reset() { Reset(nullptr); }

  void Reset(value_type* ptr) {
    Clear();
    if (ptr == nullptr) { return; }
    ptr_ = ptr;
    PtrUtil::Ref<value_type>(ptr_);
  }

  SharedPtr& operator=(const SharedPtr& rhs) {
    Reset(rhs.ptr_);
    return *this;
  }

  static SharedPtr __UnsafeMove__(value_type* ptr) {
    SharedPtr ret;
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
    PtrUtil::ReleaseRef<value_type>(ptr_);
    ptr_ = nullptr;
  }
  value_type* ptr_;
};

template<typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args) {
  return SharedPtr<T>::MakeShared(std::forward<Args>(args)...);
}

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_SHARED_PTR_H_
