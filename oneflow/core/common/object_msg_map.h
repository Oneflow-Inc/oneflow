#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_MAP_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_MAP_H_

#include "oneflow/core/common/object_msg_skiplist.h"

#define OBJECT_MSG_DEFINE_MAP_KEY(T, field_name) OBJECT_MSG_DEFINE_SKIPLIST_KEY(20, T, field_name)
#define OBJECT_MSG_DEFINE_MAP_HEAD(field_type, field_name) \
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(field_type, field_name)
#define OBJECT_MSG_MAP(field_type, field_name) OBJECT_MSG_SKIPLIST(field_type, field_name)

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_MAP_H_
