#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace user_op {

#define BASIC_ATTR_SEQ_ENTRY(field, cpp_type, attr_type)                                      \
  template<>                                                                                  \
  cpp_type AttrValAccessor<cpp_type>::GetAttr(const UserOpAttrVal& val) {                     \
    return val.field();                                                                       \
  }                                                                                           \
  template<>                                                                                  \
  void AttrValAccessor<cpp_type>::SetAttr(const cpp_type& cpp_val, UserOpAttrVal* attr_val) { \
    attr_val->set_##field(cpp_val);                                                           \
  }

OF_PP_FOR_EACH_TUPLE(BASIC_ATTR_SEQ_ENTRY, BASIC_ATTR_SEQ)

#undef BASIC_ATTR_SEQ_ENTRY

template<>
Shape AttrValAccessor<Shape>::GetAttr(const UserOpAttrVal& val) {
  return Shape(val.at_shape());
}
template<>
void AttrValAccessor<Shape>::SetAttr(const Shape& cpp_val, UserOpAttrVal* attr_val) {
  cpp_val.ToProto(attr_val->mutable_at_shape());
}

#define LIST_ATTR_SEQ_ENTRY(field, cpp_type, attr_type)                                         \
  template<>                                                                                    \
  cpp_type AttrValAccessor<cpp_type>::GetAttr(const UserOpAttrVal& val) {                       \
    return PbRf2StdVec<cpp_type::value_type>(val.field().val());                                \
  }                                                                                             \
  template<>                                                                                    \
  void AttrValAccessor<cpp_type>::SetAttr(const cpp_type& cpp_val, UserOpAttrVal* attr_val) {   \
    *(attr_val->mutable_##field()->mutable_val()) = StdVec2PbRf<cpp_type::value_type>(cpp_val); \
  }

OF_PP_FOR_EACH_TUPLE(LIST_ATTR_SEQ_ENTRY, LIST_ATTR_SEQ)

#undef LIST_ATTR_SEQ_ENTRY
}  // namespace user_op

}  // namespace oneflow
