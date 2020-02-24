#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_RAW_PTR_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_RAW_PTR_H_

#include "oneflow/core/common/struct_traits.h"
#include "oneflow/core/common/object_msg_core.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_RAW_PTR(field_type, field_name)                           \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                   \
  _OBJECT_MSG_DEFINE_RAW_PTR(STATIC_COUNTER(field_counter), field_type, field_name);

// details
#define _OBJECT_MSG_DEFINE_RAW_PTR(field_counter, field_type, field_name) \
  _OBJECT_MSG_DEFINE_RAW_PTR_FIELD(field_type, field_name)                \
  OBJECT_MSG_OVERLOAD_FIELD_TYPE_ID(field_counter, field_type);           \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgRawPtrInit);           \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgRawPtrDelete);       \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define OBJECT_MSG_OVERLOAD_FIELD_TYPE_ID(field_counter, field_type)      \
 public:                                                                  \
  template<typename FieldType, typename Enable>                           \
  struct __DssFieldTypeId__<field_counter, FieldType, Enable> final {     \
    static std::string Call() { return OF_PP_STRINGIZE(field_type) "*"; } \
  };

#define _OBJECT_MSG_DEFINE_RAW_PTR_FIELD(field_type, field_name)                           \
 public:                                                                                   \
  ConstType<field_type>& field_name() const { return *OF_PP_CAT(field_name, _); }          \
  bool OF_PP_CAT(has_, field_name)() const { return OF_PP_CAT(field_name, _) != nullptr; } \
  void OF_PP_CAT(set_, field_name)(field_type * val) { OF_PP_CAT(field_name, _) = val; }   \
  void OF_PP_CAT(clear_, field_name)() { OF_PP_CAT(set_, field_name)(nullptr); }           \
  field_type* OF_PP_CAT(mut_, field_name)() { return OF_PP_CAT(field_name, _); }           \
  field_type* OF_PP_CAT(mutable_, field_name)() { return OF_PP_CAT(field_name, _); }       \
                                                                                           \
 private:                                                                                  \
  field_type* OF_PP_CAT(field_name, _);

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgRawPtrInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static_assert(std::is_pointer<PtrFieldType>::value, "PtrFieldType is not a pointer type");
    *field = nullptr;
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgRawPtrDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static_assert(std::is_pointer<PtrFieldType>::value, "PtrFieldType is not a pointer type");
  }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_RAW_PTR_H_
