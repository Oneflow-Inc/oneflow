#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_ATTR_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_ATTR_H_

#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace user_op {

// SEQ
#define BASIC_ATTR_SEQ                                               \
  OF_PP_MAKE_TUPLE_SEQ(at_int32, int32_t, UserOpAttrType::kAtInt32)  \
  OF_PP_MAKE_TUPLE_SEQ(at_int64, int64_t, UserOpAttrType::kAtInt64)  \
  OF_PP_MAKE_TUPLE_SEQ(at_bool, bool, UserOpAttrType::kAtBool)       \
  OF_PP_MAKE_TUPLE_SEQ(at_float, float, UserOpAttrType::kAtFloat)    \
  OF_PP_MAKE_TUPLE_SEQ(at_double, double, UserOpAttrType::kAtDouble) \
  OF_PP_MAKE_TUPLE_SEQ(at_string, std::string, UserOpAttrType::kAtString)

#define ENUM_ATTR_SEQ OF_PP_MAKE_TUPLE_SEQ(at_data_type, DataType, UserOpAttrType::kAtDataType)

#define MESSAGE_ATTR_SEQ OF_PP_MAKE_TUPLE_SEQ(at_shape, Shape, UserOpAttrType::kAtShape)

#define LIST_BASIC_ATTR_SEQ                                                               \
  OF_PP_MAKE_TUPLE_SEQ(at_list_int32, std::vector<int32_t>, UserOpAttrType::kAtListInt32) \
  OF_PP_MAKE_TUPLE_SEQ(at_list_int64, std::vector<int64_t>, UserOpAttrType::kAtListInt64) \
  OF_PP_MAKE_TUPLE_SEQ(at_list_float, std::vector<float>, UserOpAttrType::kAtListFloat)

#define LIST_ENUM_ATTR_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(at_list_data_type, std::vector<DataType>, UserOpAttrType::kAtListDataType)

#define LIST_MESSAGE_ATTR_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(at_list_shape, std::vector<Shape>, UserOpAttrType::kAtListShape)

#define ATTR_SEQ      \
  BASIC_ATTR_SEQ      \
  ENUM_ATTR_SEQ       \
  MESSAGE_ATTR_SEQ    \
  LIST_BASIC_ATTR_SEQ \
  LIST_ENUM_ATTR_SEQ  \
  LIST_MESSAGE_ATTR_SEQ

// Type Trait: GetAttrType

template<typename T>
struct GetAttrType;

#define SPECIALIZE_GET_ATTR_TYPE(field, type_cpp, type_proto) \
  template<>                                                  \
  struct GetAttrType<type_cpp> : std::integral_constant<UserOpAttrType, type_proto> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_ATTR_TYPE, ATTR_SEQ);
#undef SPECIALIZE_GET_ATTR_TYPE

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_ATTR_H_
