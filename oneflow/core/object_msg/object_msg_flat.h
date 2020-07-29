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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_FLAT_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_FLAT_H_

#include "oneflow/core/object_msg/object_msg_core.h"
#include "oneflow/core/object_msg/flat_msg.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_FLAT_MSG(field_type, field_name)                          \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  _OBJECT_MSG_DEFINE_FLAT_MSG(STATIC_COUNTER(field_counter), field_type, field_name);

// details

#define _OBJECT_MSG_DEFINE_FLAT_MSG(field_counter, field_type, field_name)       \
  _OBJECT_MSG_DEFINE_FLAT_MSG_FIELD(FLAT_MSG_TYPE_CHECK(field_type), field_name) \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgFlatMsgInit);                 \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgFlatMsgDelete);             \
  DSS_DEFINE_FIELD(field_counter, "object message", field_type, OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_FLAT_MSG_FIELD(field_type, field_name)                     \
 public:                                                                              \
  static_assert(std::is_trivial<field_type>::value,                                   \
                OF_PP_STRINGIZE(field_type) " is not trivial");                       \
  bool OF_PP_CAT(has_, field_name)() {                                                \
    return ObjectMsgFlatMsgHas<std::is_enum<field_type>::value, field_type>::Call(    \
        OF_PP_CAT(field_name, _));                                                    \
  }                                                                                   \
  const field_type& field_name() const { return OF_PP_CAT(field_name, _); }           \
  void OF_PP_CAT(clear_, field_name)() {                                              \
    ObjectMsgFlatMsgClear<std::is_enum<field_type>::value, field_type>::Call(         \
        &OF_PP_CAT(field_name, _));                                                   \
  }                                                                                   \
  field_type* OF_PP_CAT(mut_, field_name)() { return &OF_PP_CAT(field_name, _); }     \
  field_type* OF_PP_CAT(mutable_, field_name)() { return &OF_PP_CAT(field_name, _); } \
                                                                                      \
 private:                                                                             \
  field_type OF_PP_CAT(field_name, _);

template<bool is_enum, typename T>
struct ObjectMsgFlatMsgHas {
  static bool Call(const T& val) { return true; }
};

template<typename T>
struct ObjectMsgFlatMsgHas<true, T> {
  static bool Call(const T& val) { return val == static_cast<T>(0); }
};

template<bool is_enum, typename T>
struct ObjectMsgFlatMsgClear {
  static void Call(T* ptr) { ptr->clear(); }
};

template<typename T>
struct ObjectMsgFlatMsgClear<true, T> {
  static void Call(T* ptr) { *ptr = static_cast<T>(0); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgFlatMsgInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgFlatMsgDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_FLAT_H_
