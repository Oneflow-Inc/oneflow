/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_INTRUSIVE_FLAT__H_
#define ONEFLOW_CORE_INTRUSIVE_FLAT__H_

#include <array>
#include <cstring>
#include <glog/logging.h>
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/intrusive/dss.h"
#include "oneflow/core/intrusive/static_counter.h"

namespace oneflow {

#define FLAT_MSG_BEGIN(struct_name)                        \
  struct struct_name final {                               \
    using self_type = struct_name;                         \
    using self_value_type = struct_name;                   \
    static const bool __is_flat_message_type__ = true;     \
                                                           \
   public:                                                 \
    DEFINE_STATIC_COUNTER(field_counter);                  \
    DSS_BEGIN(STATIC_COUNTER(field_counter), struct_name); \
    FLAT_MSG_DEFINE_BASIC_METHODS(struct_name);            \
    FLAT_MSG_DEFINE_DEFAULT(struct_name);

#define FLAT_MSG_END(struct_name)                                                       \
  static_assert(__is_flat_message_type__, "this struct is not a flat message");         \
                                                                                        \
 public:                                                                                \
  [[maybe_unused]] static const int __NumberOfFields__ = STATIC_COUNTER(field_counter); \
                                                                                        \
 public:                                                                                \
  INCREASE_STATIC_COUNTER(field_counter);                                               \
  DSS_END(STATIC_COUNTER(field_counter), "flat message", struct_name);                  \
  }                                                                                     \
  ;

#define FLAT_MSG_DEFINE_OPTIONAL(field_type, field_name)                        \
  static_assert(__is_flat_message_type__, "this struct is not a flat message"); \
  FLAT_MSG_DEFINE_ONEOF(OF_PP_CAT(__flat_msg_optional__, field_name),           \
                        FLAT_MSG_ONEOF_FIELD(field_type, field_name))

#define FLAT_MSG_DEFINE_ONEOF(oneof_name, type_and_field_name_seq) \
  _FLAT_MSG_DEFINE_ONEOF(_FLAT_MSG_DEFINE_NOTHING, oneof_name, type_and_field_name_seq);

#define FLAT_MSG_DEFINE_STRICT_ONEOF(oneof_name, type_and_field_name_seq) \
  _FLAT_MSG_DEFINE_ONEOF(_FLAT_MSG_DEFINE_ONEOF_VALUE4TYPE, oneof_name, type_and_field_name_seq);

#define _FLAT_MSG_DEFINE_ONEOF(define_field_value4field_type, oneof_name, type_and_field_name_seq) \
  static_assert(__is_flat_message_type__, "this struct is not a flat message");                    \
  FLAT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq);                            \
  FLAT_MSG_DEFINE_ONEOF_UNION(define_field_value4field_type, oneof_name, type_and_field_name_seq); \
  FLAT_MSG_DEFINE_ONEOF_ACCESSOR(oneof_name, type_and_field_name_seq)                              \
 public:                                                                                           \
  INCREASE_STATIC_COUNTER(field_counter);                                                          \
  FLAT_MSG_DSS_DEFINE_UION_FIELD(STATIC_COUNTER(field_counter), oneof_name,                        \
                                 type_and_field_name_seq);

#define FLAT_MSG_DEFINE_REPEATED(field_type, field_name, max_size)                        \
  static_assert(__is_flat_message_type__, "this struct is not a flat message");           \
  _FLAT_MSG_DEFINE_REPEATED_FIELD(FLAT_MSG_TYPE_CHECK(field_type), field_name, max_size); \
                                                                                          \
 public:                                                                                  \
  INCREASE_STATIC_COUNTER(field_counter);                                                 \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "flat message",                         \
                   OF_PP_CAT(field_name, _RepeatedField), OF_PP_CAT(field_name, _));

#define FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP() _FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP()

#define FLAT_MSG_ONEOF_FIELD(field_type, field_name) \
  OF_PP_MAKE_TUPLE_SEQ(FLAT_MSG_TYPE_CHECK(field_type), field_name)

#define FLAT_MSG_ONEOF_CASE(oneof_name) _FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name)

#define FLAT_MSG_ONEOF_CASE_VALUE(field) _FLAT_MSG_ONEOF_ENUM_VALUE(field)

#define FLAT_MSG_ONEOF_NOT_SET_VALUE(field_type, oneof_name) \
  field_type::_FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name)

#define FLAT_MSG_TYPE_CHECK(type_name) FlatMsgSelfType<type_name>::type

// details

#define FLAT_MSG_DSS_DEFINE_UION_FIELD(field_counter, oneof_name, type_and_field_name_seq) \
  DSS_DEFINE_FIELD(field_counter, "flat message", OF_PP_CAT(oneof_name, _OneofType),       \
                   OF_PP_CAT(oneof_name, _));                                              \
  DSS_DEFINE_UNION_FIELD_VISITOR(                                                          \
      field_counter, case_,                                                                \
      OF_PP_FOR_EACH_TUPLE(FLAT_MSG_MAKE_UNION_TYPE7FIELD4CASE, type_and_field_name_seq));

#define FLAT_MSG_MAKE_UNION_TYPE7FIELD4CASE(field_type, field_name) \
  OF_PP_MAKE_TUPLE_SEQ(field_type, OF_PP_CAT(field_name, _), _FLAT_MSG_ONEOF_ENUM_VALUE(field_name))

template<typename T, typename Enabled = void>
struct FlatMsgSelfType {
  static_assert(T::__is_flat_message_type__, "T is not a flat message type");
  using type = T;
};

template<typename T>
struct FlatMsgSelfType<
    T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value>::type> {
  using type = T;
};

template<typename T>
struct FlatMsg final {
  using value_type = T;
  using self_value_type = value_type;
  FlatMsg() { msg_.clear(); }
  FlatMsg(const FlatMsg& rhs) { msg_.CopyFrom(rhs.msg_); }
  FlatMsg(const T& msg) { msg_.CopyFrom(msg); }

  const value_type& operator*() const { return msg_; }
  value_type& operator*() { return msg_; }
  const value_type* operator->() const { return &msg_; }
  value_type* operator->() { return &msg_; }

  const value_type& Get() const { return msg_; }
  value_type* Mutable() { return &msg_; }

  template<typename RhsT>
  bool operator==(const RhsT& rhs) const {
    static_assert(std::is_same<FlatMsg, RhsT>::value, "");
    return msg_ == rhs.msg_;
  }

  template<typename RhsT>
  bool operator!=(const RhsT& rhs) const {
    static_assert(std::is_same<FlatMsg, RhsT>::value, "");
    return msg_ != rhs.msg_;
  }

  template<typename RhsT>
  bool operator>=(const RhsT& rhs) const {
    static_assert(std::is_same<FlatMsg, RhsT>::value, "");
    return msg_ >= rhs.msg_;
  }

  template<typename RhsT>
  bool operator<=(const RhsT& rhs) const {
    static_assert(std::is_same<FlatMsg, RhsT>::value, "");
    return msg_ <= rhs.msg_;
  }

  template<typename RhsT>
  bool operator>(const RhsT& rhs) const {
    static_assert(std::is_same<FlatMsg, RhsT>::value, "");
    return msg_ > rhs.msg_;
  }

  template<typename RhsT>
  bool operator<(const RhsT& rhs) const {
    static_assert(std::is_same<FlatMsg, RhsT>::value, "");
    return msg_ < rhs.msg_;
  }

 private:
  union {
    value_type msg_;
  };
};

#define FLAT_MSG_DEFINE_DEFAULT(flat_msg_type_name)            \
  const flat_msg_type_name& __Default__() const {              \
    static const FlatMsg<flat_msg_type_name> default_flat_msg; \
    return default_flat_msg.Get();                             \
  }

template<typename T>
struct FlatMsgIsScalar final {
  static const bool value = std::is_arithmetic<T>::value || std::is_enum<T>::value;
};

template<bool is_scalar>
struct FlatMsgGetDefault final {
  template<typename T>
  static const T& Call(const T* val) {
    return val->__Default__();
  }
};
template<>
struct FlatMsgGetDefault<true> final {
  template<typename T>
  static const T& Call(const T* val) {
    return *val;
  }
};

#define _FLAT_MSG_ONEOF_CASE_NAME(oneof_name) OF_PP_CAT(oneof_name, _case)

#define _FLAT_MSG_ONEOF_ENUM_VALUE(field) SNAKE_TO_CAMEL(field)

#define _FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name) SNAKE_TO_CAMEL(oneof_name)

#define _FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) OF_PP_CAT(k_, OF_PP_CAT(oneof_name, _not_set))

#define FLAT_MSG_DEFINE_BASIC_METHODS(T) _FLAT_MSG_DEFINE_BASIC_METHODS(T)

#define _FLAT_MSG_DEFINE_BASIC_METHODS(T)                                           \
 public:                                                                            \
  void clear() { std::memset(reinterpret_cast<void*>(this), 0, sizeof(T)); }        \
  void CopyFrom(const self_type& rhs) {                                             \
    std::memcpy(reinterpret_cast<void*>(this), reinterpret_cast<const void*>(&rhs), \
                sizeof(self_type));                                                 \
  }

#define FLAT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq)     \
 public:                                                                         \
  enum _FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name) {                                   \
    _FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) = 0,                               \
    OF_PP_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_ENUM_CASE, type_and_field_name_seq) \
  }

#define MAKE_FLAT_MSG_ONEOF_ENUM_CASE(field_type, field_name) \
  _FLAT_MSG_ONEOF_ENUM_VALUE(field_name),

#define FLAT_MSG_DEFINE_ONEOF_ACCESSOR(oneof_name, type_and_field_name_seq)                    \
  _FLAT_MSG_DEFINE_ONEOF_CASE_ACCESSOR(oneof_name, _FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name));     \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_ACCESSOR, (_FLAT_MSG_ONEOF_ENUM_VALUE), \
                                   (oneof_name), type_and_field_name_seq)

#define MAKE_FLAT_MSG_ONEOF_ACCESSOR(get_enum_value, oneof_name, pair)                         \
 public:                                                                                       \
  const OF_PP_PAIR_FIRST(pair) & OF_PP_PAIR_SECOND(pair)() const {                             \
    if (OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))()) {                                          \
      return OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                   \
    }                                                                                          \
    return FlatMsgGetDefault<FlatMsgIsScalar<OF_PP_PAIR_FIRST(pair)>::value>::Call(            \
        &OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _));                      \
  }                                                                                            \
  bool OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))() const {                                      \
    return _FLAT_MSG_ONEOF_CASE_NAME(oneof_name)() == get_enum_value(OF_PP_PAIR_SECOND(pair)); \
  }                                                                                            \
  void OF_PP_CAT(clear_, OF_PP_PAIR_SECOND(pair))() {                                          \
    if (!OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))()) { return; }                               \
    OF_PP_CAT(set_, _FLAT_MSG_ONEOF_CASE_NAME(oneof_name))                                     \
    (_FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name));                                               \
  }                                                                                            \
  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mut_, OF_PP_PAIR_SECOND(pair))() {                        \
    OF_PP_CAT(set_, _FLAT_MSG_ONEOF_CASE_NAME(oneof_name))                                     \
    (get_enum_value(OF_PP_PAIR_SECOND(pair)));                                                 \
    return &OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                    \
  }                                                                                            \
  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() {                    \
    OF_PP_CAT(set_, _FLAT_MSG_ONEOF_CASE_NAME(oneof_name))                                     \
    (get_enum_value(OF_PP_PAIR_SECOND(pair)));                                                 \
    return &OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                    \
  }                                                                                            \
  void OF_PP_CAT(set_, OF_PP_PAIR_SECOND(pair))(const OF_PP_PAIR_FIRST(pair) & val) {          \
    *OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() = val;                                     \
  }

#define FLAT_MSG_DEFINE_ONEOF_UNION(define_field_value4field_type, oneof_name,             \
                                    type_and_field_name_seq)                               \
 public:                                                                                   \
  struct OF_PP_CAT(oneof_name, _OneofType) {                                               \
   public:                                                                                 \
    using self_oneof_type = OF_PP_CAT(oneof_name, _OneofType);                             \
    using self_oneof_case_type = _FLAT_MSG_ONEOF_ENUM_TYPE(oneof_name);                    \
    template<self_oneof_case_type oneof_case, typename Enabled = void>                     \
    struct FieldType4FieldValueStruct {};                                                  \
    template<self_oneof_case_type oneof_case, typename Enabled = void>                     \
    struct HasStruct {};                                                                   \
    template<self_oneof_case_type oneof_case, typename Enabled = void>                     \
    struct GetStruct {};                                                                   \
    template<self_oneof_case_type oneof_case, typename Enabled = void>                     \
    struct MutableStruct {};                                                               \
    OF_PP_FOR_EACH_TUPLE(_MAKE_FLAT_MSG_ONEOF_TEMPLATE_ACCESSOR, type_and_field_name_seq); \
    define_field_value4field_type(type_and_field_name_seq);                                \
    template<self_oneof_case_type oneof_case>                                              \
    bool Has() const {                                                                     \
      return HasStruct<oneof_case>::Call(*this);                                           \
    }                                                                                      \
    template<self_oneof_case_type oneof_case>                                              \
    const typename FieldType4FieldValueStruct<oneof_case>::type& Get() const {             \
      return GetStruct<oneof_case>::Call(*this);                                           \
    }                                                                                      \
    template<self_oneof_case_type oneof_case>                                              \
    typename FieldType4FieldValueStruct<oneof_case>::type* Mutable() {                     \
      return MutableStruct<oneof_case>::Call(this);                                        \
    }                                                                                      \
                                                                                           \
    union {                                                                                \
      OF_PP_FOR_EACH_TUPLE(MAKE_FLAT_MSG_ONEOF_UNION_FIELD, type_and_field_name_seq)       \
    };                                                                                     \
    self_oneof_case_type case_;                                                            \
  };                                                                                       \
                                                                                           \
 private:                                                                                  \
  OF_PP_CAT(oneof_name, _OneofType) OF_PP_CAT(oneof_name, _);                              \
                                                                                           \
 public:                                                                                   \
  const OF_PP_CAT(oneof_name, _OneofType) & oneof_name() const {                           \
    return OF_PP_CAT(oneof_name, _);                                                       \
  }                                                                                        \
  OF_PP_CAT(oneof_name, _OneofType) * OF_PP_CAT(mutable_, oneof_name)() {                  \
    return &OF_PP_CAT(oneof_name, _);                                                      \
  }

#define _MAKE_FLAT_MSG_ONEOF_TEMPLATE_ACCESSOR(field_type, field_name)                 \
 public:                                                                               \
  template<typename Enabled>                                                           \
  struct FieldType4FieldValueStruct<_FLAT_MSG_ONEOF_ENUM_VALUE(field_name), Enabled> { \
    using type = field_type;                                                           \
  };                                                                                   \
  template<typename Enabled>                                                           \
  struct HasStruct<_FLAT_MSG_ONEOF_ENUM_VALUE(field_name), Enabled> {                  \
    static bool Call(const self_oneof_type& self) {                                    \
      return self.case_ == _FLAT_MSG_ONEOF_ENUM_VALUE(field_name);                     \
    }                                                                                  \
  };                                                                                   \
  template<typename Enabled>                                                           \
  struct GetStruct<_FLAT_MSG_ONEOF_ENUM_VALUE(field_name), Enabled> {                  \
    static const field_type& Call(const self_oneof_type& self) {                       \
      return self.OF_PP_CAT(field_name, _);                                            \
    }                                                                                  \
  };                                                                                   \
  template<typename Enabled>                                                           \
  struct MutableStruct<_FLAT_MSG_ONEOF_ENUM_VALUE(field_name), Enabled> {              \
    static field_type* Call(self_oneof_type* self) {                                   \
      self->case_ = _FLAT_MSG_ONEOF_ENUM_VALUE(field_name);                            \
      return &self->OF_PP_CAT(field_name, _);                                          \
    }                                                                                  \
  };

#define _FLAT_MSG_DEFINE_NOTHING(type_and_field_name_seq)

#define _FLAT_MSG_DEFINE_ONEOF_VALUE4TYPE(type_and_field_name_seq)                \
 public:                                                                          \
  template<typename T, typename Enabled = void>                                   \
  struct FieldValue4FieldType {};                                                 \
  OF_PP_FOR_EACH_TUPLE(_MAKE_FLAT_MSG_ONEOF_VALUE4TYPE, type_and_field_name_seq); \
  template<typename T>                                                            \
  bool HasField() const {                                                         \
    return Has<FieldValue4FieldType<T>::value>();                                 \
  }                                                                               \
  template<typename T>                                                            \
  const T& GetField() const {                                                     \
    return Get<FieldValue4FieldType<T>::value>();                                 \
  }                                                                               \
  template<typename T>                                                            \
  T* MutableField() {                                                             \
    return Mutable<FieldValue4FieldType<T>::value>();                             \
  }

#define _MAKE_FLAT_MSG_ONEOF_VALUE4TYPE(field_type, field_name)                       \
  template<typename Enabled>                                                          \
  struct FieldValue4FieldType<field_type, Enabled> {                                  \
    static const self_oneof_case_type value = _FLAT_MSG_ONEOF_ENUM_VALUE(field_name); \
  };

#define MAKE_FLAT_MSG_ONEOF_UNION_FIELD(field_type, field_name) field_type OF_PP_CAT(field_name, _);

#define SNAKE_TO_CAMEL(name) OF_PP_CAT(__FlatMsgSnakeToCamel__, name)

#define _FLAT_MSG_DEFINE_ONEOF_CASE_ACCESSOR(oneof_name, T)                             \
 public:                                                                                \
  T OF_PP_CAT(oneof_name, _case)() const { return OF_PP_CAT(oneof_name, _).case_; }     \
  bool OF_PP_CAT(has_, oneof_name)() const {                                            \
    return OF_PP_CAT(oneof_name, _).case_ != _FLAT_MSG_ONEOF_NOT_SET_VALUE(oneof_name); \
  }                                                                                     \
                                                                                        \
 private:                                                                               \
  void OF_PP_CAT(set_, OF_PP_CAT(oneof_name, _case))(T val) {                           \
    OF_PP_CAT(oneof_name, _).case_ = val;                                               \
  }

#define _FLAT_MSG_DEFINE_REPEATED_FIELD(T, field_name, N)                                       \
 public:                                                                                        \
  using OF_PP_CAT(field_name, _RepeatedField) = FlatMsgRepeatedField<T, N>;                     \
  std::size_t OF_PP_CAT(field_name, _size)() const { return OF_PP_CAT(field_name, _).size(); }  \
  const OF_PP_CAT(field_name, _RepeatedField) & field_name() const {                            \
    return OF_PP_CAT(field_name, _);                                                            \
  }                                                                                             \
  const T& field_name(int32_t i) const { return OF_PP_CAT(field_name, _).Get(i); }              \
  OF_PP_CAT(field_name, _RepeatedField) * OF_PP_CAT(mut_, field_name)() {                       \
    return &OF_PP_CAT(field_name, _);                                                           \
  }                                                                                             \
  OF_PP_CAT(field_name, _RepeatedField) * OF_PP_CAT(mutable_, field_name)() {                   \
    return &OF_PP_CAT(field_name, _);                                                           \
  }                                                                                             \
  T* OF_PP_CAT(mut_, field_name)(int32_t i) { return OF_PP_CAT(field_name, _).Mutable(i); }     \
  T* OF_PP_CAT(mutable_, field_name)(int32_t i) { return OF_PP_CAT(field_name, _).Mutable(i); } \
  T* OF_PP_CAT(add_, field_name)() { return OF_PP_CAT(field_name, _).Add(); }                   \
  void OF_PP_CAT(clear_, field_name)() { OF_PP_CAT(field_name, _).clear(); }                    \
                                                                                                \
 private:                                                                                       \
  OF_PP_CAT(field_name, _RepeatedField)                                                         \
  OF_PP_CAT(field_name, _);

#define _FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP()                                           \
 public:                                                                                         \
  bool operator<(const self_type& rhs) const {                                                   \
    return std::memcmp(reinterpret_cast<const void*>(this), reinterpret_cast<const void*>(&rhs), \
                       sizeof(self_type))                                                        \
           < 0;                                                                                  \
  }                                                                                              \
  bool operator<=(const self_type& rhs) const {                                                  \
    return std::memcmp(reinterpret_cast<const void*>(this), reinterpret_cast<const void*>(&rhs), \
                       sizeof(self_type))                                                        \
           <= 0;                                                                                 \
  }                                                                                              \
  bool operator==(const self_type& rhs) const {                                                  \
    return std::memcmp(reinterpret_cast<const void*>(this), reinterpret_cast<const void*>(&rhs), \
                       sizeof(self_type))                                                        \
           == 0;                                                                                 \
  }                                                                                              \
  bool operator!=(const self_type& rhs) const {                                                  \
    return std::memcmp(reinterpret_cast<const void*>(this), reinterpret_cast<const void*>(&rhs), \
                       sizeof(self_type))                                                        \
           != 0;                                                                                 \
  }                                                                                              \
  bool operator>(const self_type& rhs) const {                                                   \
    return std::memcmp(reinterpret_cast<const void*>(this), reinterpret_cast<const void*>(&rhs), \
                       sizeof(self_type))                                                        \
           > 0;                                                                                  \
  }                                                                                              \
  bool operator>=(const self_type& rhs) const {                                                  \
    return std::memcmp(reinterpret_cast<const void*>(this), reinterpret_cast<const void*>(&rhs), \
                       sizeof(self_type))                                                        \
           >= 0;                                                                                 \
  }

template<typename T, std::size_t N>
class FlatMsgRepeatedField final {
 public:
  using value_type = T;
  static const int capacity = N;

  bool empty() const { return size_ == 0; }

  std::size_t size() const { return size_; }

  void clear() { size_ = 0; }

  T* begin() { return &data_[0]; }
  T* end() {
    CHECK_GE(size_, 0);
    CHECK_LE(size_, N);
    return &data_[size_];
  }

  const T* begin() const { return &data_[0]; }
  const T* end() const {
    CHECK_GE(size_, 0);
    CHECK_LE(size_, N);
    return &data_[size_];
  }

  const T& Get(int32_t index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, N);
    return data_[index];
  }

  T* Mutable(int32_t index) {
    CHECK_GE(index, 0);
    CHECK_LT(index, N);
    return &data_[index];
  }

  const T* data() const { return &Get(0); }
  T* data() { return Mutable(0); }
  T* mut_data() { return Mutable(0); }

  T* Add() {
    CHECK_GE(size_, 0);
    CHECK_LT(size_, N);
    return &data_[size_++];
  }

 private:
  std::size_t size_;
  std::array<T, N> data_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_FLAT__H_
