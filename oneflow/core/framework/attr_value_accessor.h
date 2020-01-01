#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_VAL_ACCESSOR_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_VAL_ACCESSOR_H_

#include "oneflow/core/framework/user_op_attr.h"

namespace oneflow {

namespace user_op {

template<typename T>
struct AttrValAccessor final {
  static T GetAttr(const UserOpAttrVal&);
  static void SetAttr(const T&, UserOpAttrVal*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_VAL_ACCESSOR_H_
