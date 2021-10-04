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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_FIELD_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_FIELD_H_
#include "oneflow/core/object_msg/struct_traits.h"
#include "oneflow/core/object_msg/object_msg_core.h"

namespace oneflow {

#define OBJECT_MSG_FIELD(field_type, field_name)                                    \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  _OBJECT_MSG_FIELD(STATIC_COUNTER(field_counter), field_type, field_name);

// details
#define _OBJECT_MSG_FIELD(field_counter, field_type, field_name)   \
 private:                                                          \
  field_type field_name;                                           \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgFieldInit);     \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgFieldDelete); \
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
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_FIELD_H_
