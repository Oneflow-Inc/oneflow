#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_RAW_PTR_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_RAW_PTR_H_

#include "oneflow/core/common/object_msg_core.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_RAW_PTR(field_type, field_name)                           \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_RAW_PTR(DSS_GET_FIELD_COUNTER(), field_type, field_name);

// details
#define _OBJECT_MSG_DEFINE_RAW_PTR(field_counter, field_type, field_name) \
  _OBJECT_MSG_DEFINE_RAW_PTR_FIELD(field_type, field_name)                \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgRawPtrInit);           \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgRawPtrDelete);       \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_RAW_PTR_FIELD(field_type, field_name)                       \
 public:                                                                               \
  static_assert(std::is_pointer<field_type>::value,                                    \
                OF_PP_STRINGIZE(field_type) "is not a pointer");                       \
  bool OF_PP_CAT(has_, field_name)() { return OF_PP_CAT(field_name, _) != nullptr; }   \
  void OF_PP_CAT(set_, field_name)(field_type val) { OF_PP_CAT(field_name, _) = val; } \
  void OF_PP_CAT(clear_, field_name)() { OF_PP_CAT(set_, field_name)(nullptr); }       \
  DSS_DEFINE_GETTER(field_type, field_name);                                           \
  DSS_DEFINE_MUTABLE(field_type, field_name);                                          \
                                                                                       \
 private:                                                                              \
  field_type OF_PP_CAT(field_name, _);

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgRawPtrInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    static_assert(std::is_pointer<PtrFieldType>::value, "PtrFieldType is not a pointer type");
    *field = nullptr;
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgRawPtrDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    static_assert(std::is_pointer<PtrFieldType>::value, "PtrFieldType is not a pointer type");
  }
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_RAW_PTR_H_
