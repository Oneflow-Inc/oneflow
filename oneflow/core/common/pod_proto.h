#ifndef ONEFLOW_CORE_COMMON_POD_PROTO_H_
#define ONEFLOW_CORE_COMMON_POD_PROTO_H_

#include "oneflow/core/common/preprocessor.h"

#define POD_PROTO_DEFINE_FIELD(type, field_name) _POD_PROTO_DEFINE_FIELD(type, field_name)

#define POD_PROTO_ONEOF_ENUM_VALUE(field) SNAKE_TO_CAMEL(field)

#define POD_PROTO_ONEOF_ENUM_TYPE(oneof_name) SNAKE_TO_CAMEL(oneof_name)

#define POD_PROTO_ONEOF_FIELD(type, field_name) OF_PP_MAKE_TUPLE_SEQ(type, field_name)

#define POD_PROTO_ONEOF_CASE(oneof_name) OF_PP_CAT(oneof_name, _case)

#define POD_PROTO_DEFINE_ONEOF(oneof_name, type_and_field_name_seq)                                \
  POD_PROTO_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq);                           \
  POD_PROTO_DEFINE_ONEOF_ACCESSOR(POD_PROTO_ONEOF_ENUM_VALUE, oneof_name, type_and_field_name_seq) \
  POD_PROTO_DEFINE_ONEOF_UNION(type_and_field_name_seq);                                           \
  POD_PROTO_DEFINE_FIELD(POD_PROTO_ONEOF_ENUM_TYPE(oneof_name), POD_PROTO_ONEOF_CASE(oneof_name));

// details

#define POD_PROTO_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq)     \
 public:                                                                          \
  enum POD_PROTO_ONEOF_ENUM_TYPE(oneof_name) {                                    \
    OF_PP_FOR_EACH_TUPLE(MAKE_POD_PROTO_ONEOF_ENUM_CASE, type_and_field_name_seq) \
  }

#define MAKE_POD_PROTO_ONEOF_ENUM_CASE(type, field_name) POD_PROTO_ONEOF_ENUM_VALUE(field_name),

#define POD_PROTO_DEFINE_ONEOF_ACCESSOR(get_enum_value, oneof_name, type_and_field_name_seq)      \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_POD_PROTO_ONEOF_ACCESSOR, (get_enum_value), (oneof_name), \
                                   type_and_field_name_seq)

#define MAKE_POD_PROTO_ONEOF_ACCESSOR(get_enum_value, oneof_name, pair)                   \
  const OF_PP_PAIR_FIRST(pair) & OF_PP_PAIR_SECOND(pair)() const {                        \
    CHECK(POD_PROTO_ONEOF_CASE(oneof_name)() == get_enum_value(OF_PP_PAIR_SECOND(pair))); \
    return OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                                         \
  }                                                                                       \
  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() {               \
    OF_PP_CAT(set_, POD_PROTO_ONEOF_CASE(oneof_name))                                     \
    (get_enum_value(OF_PP_PAIR_SECOND(pair)));                                            \
    return &OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                                        \
  }

#define POD_PROTO_DEFINE_ONEOF_UNION(type_and_field_name_seq)                       \
 private:                                                                           \
  union {                                                                           \
    OF_PP_FOR_EACH_TUPLE(MAKE_POD_PROTO_ONEOF_UNION_FIELD, type_and_field_name_seq) \
  };

#define MAKE_POD_PROTO_ONEOF_UNION_FIELD(type, field_name) type OF_PP_CAT(field_name, _);

#define SNAKE_TO_CAMEL(name) OF_PP_CAT(__OneflowSnakeToCamel__, name)

#define _POD_PROTO_DEFINE_FIELD(type, field_name)                                \
 public:                                                                         \
  type field_name() const { return OF_PP_CAT(field_name, _); }                   \
  void OF_PP_CAT(set_, field_name)(type val) { OF_PP_CAT(field_name, _) = val; } \
                                                                                 \
 private:                                                                        \
  type OF_PP_CAT(field_name, _);

#endif  // ONEFLOW_CORE_COMMON_POD_PROTO_H_
