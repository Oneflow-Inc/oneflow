#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_ATTR_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_ATTR_H_

#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

// SEQ
#define ATTR_TYPE_SEQ                                                      \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, UserOpAttrType::kAtInt32)                  \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, UserOpAttrType::kAtInt64)                  \
  OF_PP_MAKE_TUPLE_SEQ(bool, UserOpAttrType::kAtBool)                      \
  OF_PP_MAKE_TUPLE_SEQ(float, UserOpAttrType::kAtFloat)                    \
  OF_PP_MAKE_TUPLE_SEQ(double, UserOpAttrType::kAtDouble)                  \
  OF_PP_MAKE_TUPLE_SEQ(std::string, UserOpAttrType::kAtString)             \
  OF_PP_MAKE_TUPLE_SEQ(Shape, UserOpAttrType::kAtShape)                    \
  OF_PP_MAKE_TUPLE_SEQ(std::vector<int32_t>, UserOpAttrType::kAtListInt32) \
  OF_PP_MAKE_TUPLE_SEQ(std::vector<int64_t>, UserOpAttrType::kAtListInt64) \
  OF_PP_MAKE_TUPLE_SEQ(std::vector<float>, UserOpAttrType::kAtListFloat)

// Type Trait: GetAttrType

template<typename T>
struct GetAttrType;

#define SPECIALIZE_GET_ATTR_TYPE(type_cpp, type_proto) \
  template<>                                           \
  struct GetAttrType<type_cpp> : std::integral_constant<UserOpAttrType, type_proto> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_ATTR_TYPE, ATTR_TYPE_SEQ);
#undef SPECIALIZE_GET_ATTR_TYPE

// Others
template<typename VecT, typename ListAttrT>
void SerializeVector2ListAttr(const VecT& vec, ListAttrT* attr) {
  // TODO(niuchong): some check for VecT and ListAttrT
  attr->clear_val();
  for (auto it = vec.begin(); it != vec.end(); ++it) { attr->add_val(*it); }
}
}

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_ATTR_H_
