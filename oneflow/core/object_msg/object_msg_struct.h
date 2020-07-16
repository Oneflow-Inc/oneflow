#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_STRUCT_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_STRUCT_H_

#include "oneflow/core/object_msg/struct_traits.h"
#include "oneflow/core/object_msg/object_msg_core.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_STRUCT(field_type, field_name)                            \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                   \
  _OBJECT_MSG_DEFINE_STRUCT(STATIC_COUNTER(field_counter), field_type, field_name);

// details
#define _OBJECT_MSG_DEFINE_STRUCT(field_counter, field_type, field_name)                        \
  _OBJECT_MSG_DEFINE_STRUCT_FIELD(field_type, field_name)                                       \
  OBJECT_MSG_OVERLOAD_FIELD_TYPE_ID(field_counter, field_type);                                 \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgStructInit);                                 \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgStructDelete);                             \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _ObjectMsgFieldType), \
                   OF_PP_CAT(field_name, _));

#define OBJECT_MSG_OVERLOAD_FIELD_TYPE_ID(field_counter, field_type)      \
 public:                                                                  \
  template<typename FieldType, typename Enable>                           \
  struct __DssFieldTypeId__<field_counter, FieldType, Enable> final {     \
    static std::string Call() { return OF_PP_STRINGIZE(field_type) "*"; } \
  };

#define _OBJECT_MSG_DEFINE_STRUCT_FIELD(field_type, field_name)                                 \
 public:                                                                                        \
  ConstType<field_type>& field_name() const {                                                   \
    ConstType<field_type>* __attribute__((__may_alias__)) ptr =                                 \
        reinterpret_cast<ConstType<field_type>*>(&OF_PP_CAT(field_name, _).data[0]);            \
    return *ptr;                                                                                \
  }                                                                                             \
  field_type* OF_PP_CAT(mutable_, field_name)() { return OF_PP_CAT(field_name, _).mut_data(); } \
  field_type* OF_PP_CAT(mut_, field_name)() { return OF_PP_CAT(mutable_, field_name)(); }       \
  template<typename T>                                                                          \
  void OF_PP_CAT(set_, field_name)(T val) {                                                     \
    static_assert(std::is_scalar<T>::value, "only scalar data type supported");                 \
    *OF_PP_CAT(mut_, field_name)() = val;                                                       \
  }                                                                                             \
                                                                                                \
 private:                                                                                       \
  union OF_PP_CAT(field_name, _ObjectMsgFieldType) {                                            \
   public:                                                                                      \
    using data_type = field_type;                                                               \
    data_type* mut_data() {                                                                     \
      data_type* __attribute__((__may_alias__)) ptr = reinterpret_cast<data_type*>(&data[0]);   \
      return ptr;                                                                               \
    }                                                                                           \
    char data[sizeof(data_type)];                                                               \
    int64_t alignment_;                                                                         \
  } OF_PP_CAT(field_name, _);

template<typename WalkCtxType, typename FieldType>
struct ObjectMsgStructInit {
  static void Call(WalkCtxType* ctx, FieldType* field) {
    using data_type = typename FieldType::data_type;
    new (&field->data[0]) data_type();
  }
};

template<typename WalkCtxType, typename FieldType>
struct ObjectMsgStructDelete {
  static void Call(WalkCtxType* ctx, FieldType* field) {
    using data_type = typename FieldType::data_type;
    field->mut_data()->data_type::~data_type();
  }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_STRUCT_H_
