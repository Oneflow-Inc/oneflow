#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_MAP_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_MAP_H_

#include "oneflow/core/common/object_msg_skiplist.h"

#define OBJECT_MSG_DEFINE_MAP_KEY(T, field_name) OBJECT_MSG_DEFINE_SKIPLIST_KEY(20, T, field_name)
#define OBJECT_MSG_DEFINE_MAP_HEAD(elem_type, elem_field_name, field_name) \
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(elem_type, elem_field_name, field_name)
#define OBJECT_MSG_MAP(elem_type, elem_field_name) OBJECT_MSG_SKIPLIST(elem_type, elem_field_name)

#define OBJECT_MSG_MAP_FOR_EACH(skiplist_ptr, elem) OBJECT_MSG_SKIPLIST_FOR_EACH(skiplist_ptr, elem)
#define OBJECT_MSG_MAP_FOR_EACH_PTR(skiplist_ptr, elem) \
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(skiplist_ptr, elem)
#define OBJECT_MSG_MAP_UNSAFE_FOR_EACH_PTR(skiplist_ptr, elem) \
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(skiplist_ptr, elem)

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_MAP_H_
