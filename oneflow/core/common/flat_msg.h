#ifndef ONEFLOW_CORE_COMMON_FLAT_MSG_H_
#define ONEFLOW_CORE_COMMON_FLAT_MSG_H_

#include <array>
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/struct_traits.h"

namespace oneflow {

#define BEGIN_FLAT_MSG(struct_name)                         \
  struct FLAT_MSG_TYPE(struct_name) final {                 \
    using this_pointer_type = FLAT_MSG_TYPE(struct_name) *; \
    DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET();          \
    FLAT_MSG_DEFINE_BASIC_METHODS(FLAT_MSG_TYPE(struct_name));

#define END_FLAT_MSG(struct_name)                                            \
  DSS_STATIC_ASSERT_STRUCT_SIZE("flat message", FLAT_MSG_TYPE(struct_name)); \
  }                                                                          \
  ;

#define FLAT_MSG(struct_name) FlatMsg<FLAT_MSG_TYPE(struct_name)>

#define FLAT_MSG_DEFINE_FIELD(type, field_name)                       \
  FLAT_MSG_DEFINE_ONEOF(OF_PP_CAT(__flat_msg_optional__, field_name), \
                        FLAT_MSG_ONEOF_FIELD(type, field_name))

#define FLAT_MSG_DEFINE_ONEOF(oneof_name, type_and_field_name_seq)                                 \
  FLAT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq);                            \
  FLAT_MSG_DEFINE_ONEOF_ACCESSOR(oneof_name, MAKE_FLAT_MSG_TYPE_SEQ(type_and_field_name_seq))      \
  FLAT_MSG_DEFINE_ONEOF_UNION(oneof_name, MAKE_FLAT_MSG_TYPE_SEQ(type_and_field_name_seq));        \
  _FLAT_MSG_DEFINE_FIELD(_FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name), _FLAT_MSG_ONEOF_CASE(oneof_name)); \
  DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET(                                                       \
      "flat message",                                                                              \
      (sizeof(((this_pointer_type) nullptr)->OF_PP_CAT(oneof_name, _))                             \
       + sizeof(((this_pointer_type) nullptr)->OF_PP_CAT(_FLAT_MSG_ONEOF_CASE(oneof_name), _))),   \
      (&((this_pointer_type) nullptr)->OF_PP_CAT(oneof_name, _)));

#define FLAT_MSG_DEFINE_REPEATED_FIELD(type, field_name, max_size)                     \
  _FLAT_MSG_DEFINE_REPEATED_FIELD(FLAT_MSG_TYPE(type), field_name, max_size);          \
  DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET(                                           \
      "flat message", sizeof(((this_pointer_type) nullptr)->OF_PP_CAT(field_name, _)), \
      (&((this_pointer_type) nullptr)->OF_PP_CAT(field_name, _)));

#define FLAT_MSG_ONEOF_FIELD(type, field_name) OF_PP_MAKE_TUPLE_SEQ(type, field_name)

#define FLAT_MSG_ONEOF_ENUM_TYPE(type, oneof_name) \
  FLAT_MSG_TYPE(type)::_FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name)

#define FLAT_MSG_ONEOF_ENUM_VALUE(type, field) \
  FLAT_MSG_TYPE(type)::_FLAT_MSG_ONEOF_ENUM_VALUE(field)

#define FLAT_MSG_ONEOF_NOT_SET_VALUE(type, oneof_name) \
  FLAT_MSG_TYPE(type)::_FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name)

#define FLAT_MSG_TYPE(type_name) OF_PP_CAT(type_name, __flat_msg_type__)

// details

template<typename T>
struct FlatMsg final {
  FlatMsg() { msg_.clear(); }

  const T& Get() const { return msg_; }
  T* Mutable() { return &msg_; }

 private:
  union {
    T msg_;
  };
};

#define DEFINE_FLAT_MSG_TYPE(type_name) typedef type_name FLAT_MSG_TYPE(type_name)
DEFINE_FLAT_MSG_TYPE(char);
DEFINE_FLAT_MSG_TYPE(int8_t);
DEFINE_FLAT_MSG_TYPE(uint8_t);
DEFINE_FLAT_MSG_TYPE(int16_t);
DEFINE_FLAT_MSG_TYPE(uint16_t);
DEFINE_FLAT_MSG_TYPE(int32_t);
DEFINE_FLAT_MSG_TYPE(uint32_t);
DEFINE_FLAT_MSG_TYPE(int64_t);
DEFINE_FLAT_MSG_TYPE(uint64_t);
DEFINE_FLAT_MSG_TYPE(float);
DEFINE_FLAT_MSG_TYPE(double);

#define _FLAT_MSG_ONEOF_CASE(oneof_name) OF_PP_CAT(oneof_name, _case)

#define _FLAT_MSG_ONEOF_ENUM_VALUE(field) SNAKE_TO_CAMEL(field)

#define _FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name) SNAKE_TO_CAMEL(oneof_name)

#define _FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) OF_PP_CAT(k_, OF_PP_CAT(oneof_name, _not_set))

#define MAKE_FLAT_MSG_TYPE_SEQ(type_and_field_name_seq) \
  OF_PP_FOR_EACH_TUPLE(SUBSTITUTE_FLAT_MSG_TYPE, type_and_field_name_seq)

#define SUBSTITUTE_FLAT_MSG_TYPE(type, field_name) \
  OF_PP_MAKE_TUPLE_SEQ(FLAT_MSG_TYPE(type), field_name)

#define FLAT_MSG_DEFINE_BASIC_METHODS(T) _FLAT_MSG_DEFINE_BASIC_METHODS(T)

#define _FLAT_MSG_DEFINE_BASIC_METHODS(T)                                                       \
 public:                                                                                        \
  void clear() { std::memset(reinterpret_cast<void*>(this), 0, sizeof(T)); }                    \
  void operator=(const T& rhs) {                                                                \
    std::memcpy(reinterpret_cast<void*>(this), reinterpret_cast<const void*>(&rhs), sizeof(T)); \
  }

#define FLAT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq)     \
 public:                                                                         \
  enum _FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name) {                                   \
    _FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) = 0,                               \
    OF_PP_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_ENUM_CASE, type_and_field_name_seq) \
  }

#define MAKE_FLAT_MSG_ONEOF_ENUM_CASE(type, field_name) _FLAT_MSG_ONEOF_ENUM_VALUE(field_name),

#define FLAT_MSG_DEFINE_ONEOF_ACCESSOR(oneof_name, type_and_field_name_seq)                    \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_ACCESSOR, (_FLAT_MSG_ONEOF_ENUM_VALUE), \
                                   (oneof_name), type_and_field_name_seq)

#define MAKE_FLAT_MSG_ONEOF_ACCESSOR(get_enum_value, oneof_name, pair)                    \
 public:                                                                                  \
  const OF_PP_PAIR_FIRST(pair) & OF_PP_PAIR_SECOND(pair)() const {                        \
    CHECK(_FLAT_MSG_ONEOF_CASE(oneof_name)() == get_enum_value(OF_PP_PAIR_SECOND(pair))); \
    return OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                \
  }                                                                                       \
  bool OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))() const {                                 \
    return _FLAT_MSG_ONEOF_CASE(oneof_name)() == get_enum_value(OF_PP_PAIR_SECOND(pair)); \
  }                                                                                       \
  void OF_PP_CAT(clear_, OF_PP_PAIR_SECOND(pair))() {                                     \
    OF_PP_CAT(set_, _FLAT_MSG_ONEOF_CASE(oneof_name))                                     \
    (_FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name));                                          \
  }                                                                                       \
  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() {               \
    OF_PP_CAT(set_, _FLAT_MSG_ONEOF_CASE(oneof_name))                                     \
    (get_enum_value(OF_PP_PAIR_SECOND(pair)));                                            \
    return &OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);               \
  }                                                                                       \
  void OF_PP_CAT(set_, OF_PP_PAIR_SECOND(pair))(const OF_PP_PAIR_FIRST(pair) & val) {     \
    *OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() = val;                                \
  }

#define FLAT_MSG_DEFINE_ONEOF_UNION(oneof_name, type_and_field_name_seq)           \
 private:                                                                          \
  union {                                                                          \
    OF_PP_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_UNION_FIELD, type_and_field_name_seq) \
  } OF_PP_CAT(oneof_name, _);

#define MAKE_FLAT_MSG_ONEOF_UNION_FIELD(type, field_name) type OF_PP_CAT(field_name, _);

#define SNAKE_TO_CAMEL(name) OF_PP_CAT(__FlatMsgSnakeToCamel__, name)

#define _FLAT_MSG_DEFINE_FIELD(T, field_name)                                        \
 public:                                                                             \
  const T& field_name() const { return OF_PP_CAT(field_name, _); }                   \
                                                                                     \
 private:                                                                            \
  void OF_PP_CAT(set_, field_name)(const T& val) { OF_PP_CAT(field_name, _) = val; } \
  T OF_PP_CAT(field_name, _);

#define _FLAT_MSG_DEFINE_REPEATED_FIELD(T, field_name, N)                                          \
 public:                                                                                           \
  std::size_t OF_PP_CAT(field_name, _size)() const { return OF_PP_CAT(field_name, _).size(); }     \
  const FlatMsgRepeatedField<T, N>& field_name() const { return OF_PP_CAT(field_name, _); }        \
  const T& field_name(int32_t i) const { return OF_PP_CAT(field_name, _).Get(i); }                 \
  FlatMsgRepeatedField<T, N>* OF_PP_CAT(mutable_, field_name)() {                                  \
    return &OF_PP_CAT(field_name, _);                                                              \
  }                                                                                                \
  T* OF_PP_CAT(mutable_, field_name)(int32_t i) { return OF_PP_CAT(field_name, _).Mutable(i); }    \
  void OF_PP_CAT(add_, field_name)(const T& val) { return *OF_PP_CAT(field_name, _).Add() = val; } \
  void OF_PP_CAT(clear_, field_name)() { OF_PP_CAT(field_name, _).clear(); }                       \
                                                                                                   \
 private:                                                                                          \
  FlatMsgRepeatedField<T, N> OF_PP_CAT(field_name, _)

template<typename T, std::size_t N>
struct FlatMsgRepeatedField final {
  std::size_t size() const { return size_; }

  void clear() { size_ = 0; }

  T* begin() { return &array_[0]; }
  T* end() {
    CHECK_GE(size_, 0);
    CHECK_LE(size_, N);
    return &array_[size_];
  }

  const T* begin() const { return &array_[0]; }
  const T* end() const {
    CHECK_GE(size_, 0);
    CHECK_LE(size_, N);
    return &array_[size_];
  }

  const T& Get(int32_t index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, N);
    return array_[index];
  }

  T* Mutable(int32_t index) {
    CHECK_GE(index, 0);
    CHECK_LT(index, N);
    return &array_[index];
  }

  T* Add() {
    CHECK_GE(size_, 0);
    CHECK_LT(size_, N);
    return &array_[size_++];
  }

 private:
  std::size_t size_;
  std::array<T, N> array_;
};
}

#endif  // ONEFLOW_CORE_COMMON_FLAT_MSG_H_
