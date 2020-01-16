#ifndef ONEFLOW_CORE_COMMON_FLAT_MSG_H_
#define ONEFLOW_CORE_COMMON_FLAT_MSG_H_

#include <array>
#include "oneflow/core/common/preprocessor.h"

template<typename T>
struct EmbeddedFlatMsg final {
  const T& Get() const {
    const T* __attribute__((__may_alias__)) ptr = reinterpret_cast<const T*>(&msg_[0]);
    return *ptr;
  }
  T* Mutable() {
    T* __attribute__((__may_alias__)) ptr = reinterpret_cast<T*>(&msg_[0]);
    return ptr;
  }

 private:
  char msg_[sizeof(T)];
};

// all flat message objects with class T should be instantiated by FlatMsg<T>
template<typename T>
struct FlatMsg final {
  FlatMsg() { Mutable()->clear(); }

  const T& Get() const { return msg_.Get(); }
  T* Mutable() { return msg_.Mutable(); }

 private:
  EmbeddedFlatMsg<T> msg_;
};

#define FLAT_MSG(T) _FLAT_MSG(T)

#define FLAT_MSG_DEFINE_FIELD(type, field_name)                       \
  FLAT_MSG_DEFINE_ONEOF(OF_PP_CAT(__flat_msg_optional__, field_name), \
                        FLAT_MSG_ONEOF_FIELD(type, field_name))

#define FLAT_MSG_DEFINE_ONEOF(oneof_name, type_and_field_name_seq)                               \
  static_assert(this_struct_declared_as_a_flat_message, "this struct is not a flat message");    \
  FLAT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq);                          \
  FLAT_MSG_DEFINE_ONEOF_ACCESSOR(FLAT_MSG_ONEOF_ENUM_VALUE, oneof_name, type_and_field_name_seq) \
  FLAT_MSG_DEFINE_ONEOF_UNION(type_and_field_name_seq);                                          \
  _FLAT_MSG_DEFINE_FIELD(FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name), FLAT_MSG_ONEOF_CASE(oneof_name));

#define FLAT_MSG_DEFINE_REPEATED_FIELD(type, field_name, max_size)                            \
  static_assert(this_struct_declared_as_a_flat_message, "this struct is not a flat message"); \
  _FLAT_MSG_DEFINE_REPEATED_FIELD(type, field_name, max_size)

#define FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) OF_PP_CAT(k_, OF_PP_CAT(oneof_name, _not_set))

#define FLAT_MSG_ONEOF_ENUM_VALUE(field) SNAKE_TO_CAMEL(field)

#define FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name) SNAKE_TO_CAMEL(oneof_name)

#define FLAT_MSG_ONEOF_FIELD(type, field_name) OF_PP_MAKE_TUPLE_SEQ(type, field_name)

#define FLAT_MSG_ONEOF_CASE(oneof_name) OF_PP_CAT(oneof_name, _case)

// details

#define _FLAT_MSG(T)                                                                            \
 public:                                                                                        \
  T() = delete;                                                                                 \
  const static bool this_struct_declared_as_a_flat_message = true;                              \
  void clear() { std::memset(reinterpret_cast<void*>(this), 0, sizeof(T)); }                    \
  void operator=(const T& rhs) {                                                                \
    std::memcpy(reinterpret_cast<void*>(this), reinterpret_cast<const void*>(&rhs), sizeof(T)); \
  }

#define FLAT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq)     \
 public:                                                                         \
  enum FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name) {                                    \
    FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) = 0,                                \
    OF_PP_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_ENUM_CASE, type_and_field_name_seq) \
  }

#define MAKE_FLAT_MSG_ONEOF_ENUM_CASE(type, field_name) FLAT_MSG_ONEOF_ENUM_VALUE(field_name),

#define FLAT_MSG_DEFINE_ONEOF_ACCESSOR(get_enum_value, oneof_name, type_and_field_name_seq)      \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_ACCESSOR, (get_enum_value), (oneof_name), \
                                   type_and_field_name_seq)

#define MAKE_FLAT_MSG_ONEOF_ACCESSOR(get_enum_value, oneof_name, pair)                   \
 public:                                                                                 \
  const OF_PP_PAIR_FIRST(pair) & OF_PP_PAIR_SECOND(pair)() const {                       \
    CHECK(FLAT_MSG_ONEOF_CASE(oneof_name)() == get_enum_value(OF_PP_PAIR_SECOND(pair))); \
    return OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _).Get();                                  \
  }                                                                                      \
  bool OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))() const {                                \
    return FLAT_MSG_ONEOF_CASE(oneof_name)() == get_enum_value(OF_PP_PAIR_SECOND(pair)); \
  }                                                                                      \
  void OF_PP_CAT(clear_, OF_PP_PAIR_SECOND(pair))() {                                    \
    OF_PP_CAT(set_, FLAT_MSG_ONEOF_CASE(oneof_name))                                     \
    (FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name));                                          \
  }                                                                                      \
  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() {              \
    OF_PP_CAT(set_, FLAT_MSG_ONEOF_CASE(oneof_name))                                     \
    (get_enum_value(OF_PP_PAIR_SECOND(pair)));                                           \
    return OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _).Mutable();                              \
  }                                                                                      \
  void OF_PP_CAT(set_, OF_PP_PAIR_SECOND(pair))(const OF_PP_PAIR_FIRST(pair) & val) {    \
    *OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() = val;                               \
  }

#define FLAT_MSG_DEFINE_ONEOF_UNION(type_and_field_name_seq)                       \
 private:                                                                          \
  union {                                                                          \
    OF_PP_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_UNION_FIELD, type_and_field_name_seq) \
  };

#define MAKE_FLAT_MSG_ONEOF_UNION_FIELD(type, field_name) \
  EmbeddedFlatMsg<type> OF_PP_CAT(field_name, _);

#define SNAKE_TO_CAMEL(name) OF_PP_CAT(__FlatMsgSnakeToCamel__, name)

#define _FLAT_MSG_DEFINE_FIELD(T, field_name)                                                   \
 public:                                                                                        \
  const T& field_name() const { return OF_PP_CAT(field_name, _).Get(); }                        \
                                                                                                \
 private:                                                                                       \
  void OF_PP_CAT(set_, field_name)(const T& val) { *OF_PP_CAT(field_name, _).Mutable() = val; } \
  EmbeddedFlatMsg<T> OF_PP_CAT(field_name, _);

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

  T* begin() { return array_[0].Mutable(); }
  T* end() {
    CHECK_GE(size_, 0);
    CHECK_LE(size_, N);
    return array_[size_].Mutable();
  }

  const T* begin() const { return &array_[0].Get(); }
  const T* end() const {
    CHECK_GE(size_, 0);
    CHECK_LE(size_, N);
    return &array_[size_].Get();
  }

  const T& Get(int32_t index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, N);
    return array_[index].Get();
  }

  T* Mutable(int32_t index) {
    CHECK_GE(index, 0);
    CHECK_LT(index, N);
    return array_[index].Mutable();
  }

  T* Add() {
    CHECK_GE(size_, 0);
    CHECK_LT(size_, N);
    return array_[size_++].Mutable();
  }

 private:
  std::size_t size_;
  std::array<EmbeddedFlatMsg<T>, N> array_;
};

#endif  // ONEFLOW_CORE_COMMON_FLAT_MSG_H_
