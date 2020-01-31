#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_FLAT_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_FLAT_H_

#include "oneflow/core/common/object_msg_core.h"
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_FLAT_MSG(field_type, field_name)                          \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_FLAT_MSG(DSS_GET_FIELD_COUNTER(), field_type, field_name);

// details

#define _OBJECT_MSG_DEFINE_FLAT_MSG(field_counter, field_type, field_name) \
  _OBJECT_MSG_DEFINE_FLAT_MSG_FIELD(FLAT_MSG_TYPE(field_type), field_name) \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgFlatMsgInit);           \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgFlatMsgDelete);       \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_FLAT_MSG_FIELD(field_type, field_name)            \
 public:                                                                     \
  static_assert(field_type::__is_flat_message_type__,                        \
                OF_PP_STRINGIZE(field_type) "is not a flat message type");   \
  bool OF_PP_CAT(has_, field_name)() { return true; }                        \
  DSS_DEFINE_GETTER(field_type, field_name);                                 \
  void OF_PP_CAT(clear_, field_name)() { OF_PP_CAT(field_name, _).clear(); } \
  DSS_DEFINE_MUTABLE(field_type, field_name);                                \
                                                                             \
 private:                                                                    \
  field_type OF_PP_CAT(field_name, _);

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgFlatMsgInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgFlatMsgDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {}
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_FLAT_H_
