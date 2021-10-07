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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_CORE_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_CORE_H_

#include <cstring>
#include <atomic>
#include <memory>
#include <type_traits>
#include <set>
#include <glog/logging.h>
#include "oneflow/core/object_msg/dss.h"
#include "oneflow/core/object_msg/static_counter.h"
#include "oneflow/core/object_msg/struct_traits.h"

namespace oneflow {

#define OBJECT_MSG_BEGIN(class_name)                     \
  struct class_name final : public ObjectMsgStruct {     \
   public:                                               \
    using self_type = class_name;                        \
    static const bool __is_object_message_type__ = true; \
    OF_PRIVATE DEFINE_STATIC_COUNTER(field_counter);     \
    DSS_BEGIN(STATIC_COUNTER(field_counter), class_name);

#define OBJECT_MSG_END(class_name)                                                  \
  OBJECT_MSG_DEFINE_BASE();                                                         \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OF_PUBLIC static const int __NumberOfFields__ = STATIC_COUNTER(field_counter);    \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  DSS_END(STATIC_COUNTER(field_counter), "object message", class_name);             \
  }                                                                                 \
  ;

#define OBJECT_MSG_DEFINE_FIELD(field_type, field_name)                             \
 private:                                                                           \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  field_type field_name;                                                            \
  INCREASE_STATIC_COUNTER(field_counter);                                           \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "object message", field_type, field_name);

#define OBJECT_MSG_FIELD(struct_type, field_name)                            \
  StructField<struct_type, struct_type::OF_PP_CAT(field_name, DssFieldType), \
              struct_type::OF_PP_CAT(field_name, kDssFieldOffset)>

// Get field number by field name
// note: field numbers start from 1 instead of 0.
#define OBJECT_MSG_FIELD_NUMBER(cls, field_name) cls::OF_PP_CAT(field_name, kDssFieldFieldNumber)

// Get field type by field number
#define OBJECT_MSG_FIELD_TYPE(cls, field_number) cls::template __DssFieldType__<field_number>::type

// Get field offset by field number
#define OBJECT_MSG_FIELD_OFFSET(cls, field_number) \
  cls::template __DssFieldOffset4FieldIndex__<field_number>::value

// Get current defined field counter inside a object_msg class.
// note: not used outside OBJECT_MSG_BEGIN ... OBJECT_MSG_END
// e.g.:
// OBJECT_MSG_BEGIN(Foo);
//   static_assert(OBJECT_MSG_FIELD_COUNTER == 0, "");
//   OBJECT_MSG_DEFINE_FIELD(int64_t, a);
//   static_assert(OBJECT_MSG_FIELD_COUNTER == 1, "");
//   OBJECT_MSG_DEFINE_FIELD(int64_t, b);
//   static_assert(OBJECT_MSG_FIELD_COUNTER == 2, "");
//   OBJECT_MSG_DEFINE_FIELD(int8_t, c);
//   static_assert(OBJECT_MSG_FIELD_COUNTER == 3, "");
//   OBJECT_MSG_DEFINE_FIELD(int64_t, d);
// OBJECT_MSG_END(Foo);
#define OBJECT_MSG_FIELD_COUNTER STATIC_COUNTER(field_counter)

// details

#define OBJECT_MSG_DEFINE_BASE()                                                   \
 public:                                                                           \
  ObjectMsgBase* __mut_object_msg_base__() { return &__object_msg_base__; }        \
  int32_t ref_cnt() const { return __object_msg_base__.ref_cnt(); }                \
                                                                                   \
 private:                                                                          \
  ObjectMsgBase __object_msg_base__;                                               \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                               \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "object message", ObjectMsgBase, \
                   __object_msg_base__);

#define _OBJECT_MSG_DEFINE_FIELD(field_counter, field_type, field_name) \
 private:                                                               \
  field_type field_name;                                                \
  DSS_DEFINE_FIELD(field_counter, "object message", field_type, field_name);

struct ObjectMsgStruct {
  void __Init__() {}
  void __Delete__() {}
};

class ObjectMsgBase {
 public:
  int32_t ref_cnt() const { return ref_cnt_; }

 private:
  friend struct ObjectMsgPtrUtil;
  void InitRefCount() { ref_cnt_ = 0; }
  void IncreaseRefCount() { ref_cnt_++; }
  int32_t DecreaseRefCount() { return --ref_cnt_; }

  std::atomic<int32_t> ref_cnt_;
};

struct ObjectMsgPtrUtil final {
  template<typename T>
  static void InitRef(T** ptr) {
    *ptr = new T();
    (*ptr)->__mut_object_msg_base__()->InitRefCount();
    Ref(*ptr);
  }
  template<typename T>
  static void Ref(T* ptr) {
    ptr->__mut_object_msg_base__()->IncreaseRefCount();
  }
  template<typename T>
  static void ReleaseRef(T* ptr) {
    CHECK_NOTNULL(ptr);
    int32_t ref_cnt = ptr->__mut_object_msg_base__()->DecreaseRefCount();
    if (ref_cnt > 0) { return; }
    ptr->__Delete__();
    delete ptr;
  }
};

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
    ObjectMsgPtrUtil::InitRef(&ret.ptr_);
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

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_CORE_H_
