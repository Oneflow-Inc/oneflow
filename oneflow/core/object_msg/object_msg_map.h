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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_MAP_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_MAP_H_

#include "oneflow/core/object_msg/object_msg_skiplist.h"

#define OBJECT_MSG_DEFINE_MAP_KEY(T, field_name) OBJECT_MSG_DEFINE_SKIPLIST_KEY(20, T, field_name)
#define OBJECT_MSG_DEFINE_MAP_HEAD(elem_type, elem_field_name, field_name) \
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(elem_type, elem_field_name, field_name)
#define OBJECT_MSG_MAP(elem_type, elem_field_name) OBJECT_MSG_SKIPLIST(elem_type, elem_field_name)

#define OBJECT_MSG_MAP_FOR_EACH(skiplist_ptr, elem) OBJECT_MSG_SKIPLIST_FOR_EACH(skiplist_ptr, elem)
#define OBJECT_MSG_MAP_FOR_EACH_PTR(skiplist_ptr, elem) \
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(skiplist_ptr, elem)
#define OBJECT_MSG_MAP_UNSAFE_FOR_EACH_PTR(skiplist_ptr, elem) \
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(skiplist_ptr, elem)

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_MAP_H_
