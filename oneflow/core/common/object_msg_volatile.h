#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_VOLATILE_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_VOLATILE_H_

#include "oneflow/core/common/object_msg_core.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_VOLATILE(field_type, field_name)                          \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_VOLATILE(DSS_GET_FIELD_COUNTER(), field_type, field_name);

// details

#define _OBJECT_MSG_DEFINE_VOLATILE(field_counter, field_type, field_name) \
  _OBJECT_MSG_DEFINE_VOLATILE_FIELD(field_type, field_name)                \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgVolatileInit);          \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgVolatileDelete);      \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_VOLATILE_FIELD(field_type, field_name)                         \
 public:                                                                                  \
  static_assert(std::is_arithmetic<field_type>::value || std::is_enum<field_type>::value, \
                "only scalar field supported");                                           \
  field_type field_name() const { return OF_PP_CAT(field_name, _); }                      \
  void OF_PP_CAT(set_, field_name)(field_type val) { OF_PP_CAT(field_name, _) = val; }    \
                                                                                          \
 private:                                                                                 \
  volatile field_type OF_PP_CAT(field_name, _);

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgVolatileInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) { *field = 0; }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgVolatileDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {}
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_VOLATILE_H_
