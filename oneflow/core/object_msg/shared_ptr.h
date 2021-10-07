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

template<typename T>
class ObjectMsgPtr final {
 public:
  static_assert(T::__is_object_message_type__, "T is not a object message type");
  using value_type = T;
  ObjectMsgPtr() : ptr_(nullptr) {}
  ObjectMsgPtr(value_type* ptr) : ptr_(nullptr) { Reset(ptr); }
  ObjectMsgPtr(const ObjectMsgPtr& obj_ptr) {
    ptr_ = nullptr;
    Reset(obj_ptr.ptr_);
  }
  ObjectMsgPtr(ObjectMsgPtr&& obj_ptr) {
    ptr_ = obj_ptr.ptr_;
    obj_ptr.ptr_ = nullptr;
  }
  ~ObjectMsgPtr() { Clear(); }

  operator bool() const { return ptr_ != nullptr; }
  const value_type& Get() const { return *ptr_; }
  const value_type* operator->() const { return ptr_; }
  const value_type& operator*() const { return *ptr_; }
  bool operator==(const ObjectMsgPtr& rhs) const { return this->ptr_ == rhs.ptr_; }

  value_type* Mutable() { return ptr_; }
  value_type* operator->() { return ptr_; }
  value_type& operator*() { return *ptr_; }

  void Reset() { Reset(nullptr); }

  void Reset(value_type* ptr) {
    Clear();
    if (ptr == nullptr) { return; }
    ptr_ = ptr;
    ObjectMsgPtrUtil::Ref<value_type>(ptr_);
  }

  ObjectMsgPtr& operator=(const ObjectMsgPtr& rhs) {
    Reset(rhs.ptr_);
    return *this;
  }

  template<typename... Args>
  static ObjectMsgPtr New(Args&&... args) {
    ObjectMsgPtr ret;
    ObjectMsgPtrUtil::NewAndInitRef(&ret.ptr_);
    ret.Mutable()->__Init__(std::forward<Args>(args)...);
    return ret;
  }

  static ObjectMsgPtr __UnsafeMove__(value_type* ptr) {
    ObjectMsgPtr ret;
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
    ObjectMsgPtrUtil::ReleaseRef<value_type>(ptr_);
    ptr_ = nullptr;
  }
  value_type* ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_SHARED_PTR_H_
