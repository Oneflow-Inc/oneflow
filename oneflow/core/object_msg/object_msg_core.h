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

#define OBJECT_MSG_BEGIN(class_name)                      \
  struct class_name final : public ObjectMsgStruct {      \
   public:                                                \
    using self_type = class_name;                         \
    static const bool __is_object_message_type__ = true;  \
    OF_PRIVATE DEFINE_STATIC_COUNTER(field_counter);      \
    DSS_BEGIN(STATIC_COUNTER(field_counter), class_name); \
    OBJECT_MSG_DEFINE_INIT();                             \
    OBJECT_MSG_DEFINE_DELETE();

#define OBJECT_MSG_END(class_name)                                                  \
  OBJECT_MSG_DEFINE_BASE();                                                         \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OF_PUBLIC static const int __NumberOfFields__ = STATIC_COUNTER(field_counter);    \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  DSS_END(STATIC_COUNTER(field_counter), "object message", class_name);             \
  }                                                                                 \
  ;

#define OBJECT_MSG_DEFINE_FIELD(field_type, field_name)                             \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  _OBJECT_MSG_DEFINE_FIELD(STATIC_COUNTER(field_counter), field_type, field_name);

#define OBJECT_MSG_FIELD(struct_type, field_name)                             \
  StructField<struct_type, struct_type::OF_PP_CAT(field_name, kDssFieldType), \
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

#define OBJECT_MSG_TYPE_CHECK(class_name) ObjectMsgSelfType<class_name>::type

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

#define OBJECT_MSG_DEFINE_INIT()                                            \
 public:                                                                    \
  template<typename WalkCtxType>                                            \
  void ObjectMsg__Init__(WalkCtxType* ctx) {                                \
    this->template __WalkField__<ObjectMsgField__Init__, WalkCtxType>(ctx); \
  }                                                                         \
                                                                            \
 private:                                                                   \
  template<int field_counter, typename WalkCtxType, typename PtrFieldType>  \
  struct ObjectMsgField__Init__ : public ObjectMsgNaiveInit<WalkCtxType, PtrFieldType> {};

#define OBJECT_MSG_OVERLOAD_INIT(field_counter, init_template)            \
 private:                                                                 \
  template<typename WalkCtxType, typename PtrFieldType>                   \
  struct ObjectMsgField__Init__<field_counter, WalkCtxType, PtrFieldType> \
      : public init_template<WalkCtxType, PtrFieldType> {};

#define OBJECT_MSG_OVERLOAD_INIT_WITH_FIELD_INDEX(field_counter, init_template) \
 private:                                                                       \
  template<typename WalkCtxType, typename PtrFieldType>                         \
  struct ObjectMsgField__Init__<field_counter, WalkCtxType, PtrFieldType>       \
      : public init_template<field_counter, WalkCtxType, PtrFieldType> {};

#define OBJECT_MSG_DEFINE_DELETE()                                                \
 public:                                                                          \
  void ObjectMsg__Delete__() {                                                    \
    this->__Delete__();                                                           \
    this->template __ReverseWalkField__<ObjectMsgField__Delete__, void>(nullptr); \
  }                                                                               \
                                                                                  \
 private:                                                                         \
  template<int field_counter, typename WalkCtxType, typename PtrFieldType>        \
  struct ObjectMsgField__Delete__ : public ObjectMsgNaiveDelete<WalkCtxType, PtrFieldType> {};

#define OBJECT_MSG_OVERLOAD_DELETE(field_counter, delete_template)          \
 private:                                                                   \
  template<typename WalkCtxType, typename PtrFieldType>                     \
  struct ObjectMsgField__Delete__<field_counter, WalkCtxType, PtrFieldType> \
      : public delete_template<WalkCtxType, PtrFieldType> {};

#define _OBJECT_MSG_DEFINE_FIELD(field_counter, field_type, field_name) \
 private:                                                               \
  field_type field_name;                                                \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgFieldInit);          \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgFieldDelete);      \
  DSS_DEFINE_FIELD(field_counter, "object message", field_type, field_name);

template<typename WalkCtxType, typename FieldType>
struct ObjectMsgFieldInit {
  static void Call(WalkCtxType* ctx, FieldType* field) { /* Do nothing */
  }
};

template<typename WalkCtxType, typename FieldType>
struct ObjectMsgFieldDelete {
  static void Call(WalkCtxType* ctx, FieldType* field) { /* Do nothing */
  }
};

template<typename T, typename Enabled = void>
struct ObjectMsgSelfType {
  static_assert(T::__is_object_message_type__, "T is not a object message type");
  using type = T;
};

template<typename T>
struct ObjectMsgSelfType<
    T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value>::type> {
  using type = T;
};

struct ObjectMsgPtrUtil;
template<typename T>
class ObjectMsgPtr;

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
  static void InitRef(T* ptr) {
    ptr->__mut_object_msg_base__()->InitRefCount();
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
    ptr->ObjectMsg__Delete__();
    delete ptr;
  }
};

template<bool is_pointer>
struct _ObjectMsgNaiveInit {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgNaiveInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static const bool is_ptr = std::is_pointer<PtrFieldType>::value;
    _ObjectMsgNaiveInit<is_ptr>::template Call<WalkCtxType, PtrFieldType>(ctx, field);
  }
};

template<>
struct _ObjectMsgNaiveInit<true> {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field_ptr) {
    static_assert(std::is_pointer<PtrFieldType>::value, "invalid use of _ObjectMsgNaiveInit");
    using FieldType = typename std::remove_pointer<PtrFieldType>::type;
    static_assert(std::is_base_of<ObjectMsgStruct, FieldType>::value,
                  "FieldType is not a subclass of ObjectMsgStruct");
    auto* ptr = new FieldType();
    *field_ptr = ptr;
    ObjectMsgPtrUtil::InitRef<FieldType>(ptr);
    ObjectMsgPtrUtil::Ref<FieldType>(ptr);
    ptr->template ObjectMsg__Init__<WalkCtxType>(ctx);
  }
};

template<bool is_pointer>
struct _ObjectMsgNaiveDelete {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgNaiveDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static const bool is_ptr = std::is_pointer<PtrFieldType>::value;
    _ObjectMsgNaiveDelete<is_ptr>::template Call<WalkCtxType, PtrFieldType>(ctx, field);
  }
};

template<>
struct _ObjectMsgNaiveDelete<true> {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static_assert(std::is_pointer<PtrFieldType>::value, "invalid use of _ObjectMsgNaiveDelete");
    using FieldType = typename std::remove_pointer<PtrFieldType>::type;
    static_assert(std::is_base_of<ObjectMsgStruct, FieldType>::value,
                  "FieldType is not a subclass of ObjectMsgStruct");
    auto* ptr = *field;
    if (ptr == nullptr) { return; }
    ObjectMsgPtrUtil::ReleaseRef<FieldType>(ptr);
  }
};

template<typename T>
class ObjectMsgPtr final {
 public:
  using value_type = typename OBJECT_MSG_TYPE_CHECK(T);
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
    ObjectMsgNaiveInit<void, value_type*>::Call(nullptr, &ret.ptr_);
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
