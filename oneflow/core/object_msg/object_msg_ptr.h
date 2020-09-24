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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_PTR_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_PTR_H_

#include "oneflow/core/object_msg/struct_traits.h"
#include "oneflow/core/object_msg/object_msg_core.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_PTR(field_type, field_name)                               \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  _OBJECT_MSG_DEFINE_PTR(STATIC_COUNTER(field_counter), field_type, field_name);

// details
#define _OBJECT_MSG_DEFINE_PTR(field_counter, field_type, field_name) \
  _OBJECT_MSG_DEFINE_PTR_FIELD(field_type, field_name)                \
  OBJECT_MSG_OVERLOAD_FIELD_TYPE_ID(field_counter, field_type);       \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgRawPtrInit);       \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgRawPtrDelete);   \
  DSS_DEFINE_FIELD(field_counter, "object message", field_type*, OF_PP_CAT(field_name, _));

#define OBJECT_MSG_OVERLOAD_FIELD_TYPE_ID(field_counter, field_type)      \
 public:                                                                  \
  template<typename FieldType, typename Enable>                           \
  struct __DssFieldTypeId__<field_counter, FieldType, Enable> final {     \
    static std::string Call() { return OF_PP_STRINGIZE(field_type) "*"; } \
  };

#define _OBJECT_MSG_DEFINE_PTR_FIELD(field_type, field_name)                               \
 public:                                                                                   \
  ConstType<field_type>& field_name() const { return *OF_PP_CAT(field_name, _); }          \
  bool OF_PP_CAT(has_, field_name)() const { return OF_PP_CAT(field_name, _) != nullptr; } \
  void OF_PP_CAT(set_, field_name)(field_type * val) { OF_PP_CAT(field_name, _) = val; }   \
  void OF_PP_CAT(clear_, field_name)() { OF_PP_CAT(set_, field_name)(nullptr); }           \
  field_type* OF_PP_CAT(mut_, field_name)() { return OF_PP_CAT(field_name, _); }           \
  field_type* OF_PP_CAT(mutable_, field_name)() { return OF_PP_CAT(field_name, _); }       \
                                                                                           \
 private:                                                                                  \
  field_type* OF_PP_CAT(field_name, _);

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgRawPtrInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static_assert(std::is_pointer<PtrFieldType>::value, "PtrFieldType is not a pointer type");
    *field = nullptr;
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgRawPtrDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static_assert(std::is_pointer<PtrFieldType>::value, "PtrFieldType is not a pointer type");
  }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_PTR_H_
