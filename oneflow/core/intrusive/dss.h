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
#ifndef ONEFLOW_CORE_INTRUSIVE_DSS_H_
#define ONEFLOW_CORE_INTRUSIVE_DSS_H_

#include <cstddef>
#include <typeinfo>
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/intrusive/struct_traits.h"

namespace oneflow {

// DSS is short for domain specific struct
#define DSS_BEGIN(field_counter, type) _DSS_BEGIN(field_counter, type)
#define DSS_DEFINE_FIELD(field_counter, dss_type, field_type, field_name) \
  _DSS_DEFINE_FIELD(field_counter, dss_type, field_type, field_name)
#define DSS_END(field_counter, dss_type, type) _DSS_END(field_counter, dss_type, type)
#define DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, field_case, type7field7case_tuple_seq) \
  _DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, field_case, type7field7case_tuple_seq)
#define DSS_GET_FIELD_COUNTER() __COUNTER__

// details

#define _DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, field_case, type7field7case_tuple_seq)   \
 private:                                                                                       \
  template<template<int, class, class> class F, typename WalkCtxType, typename DssFieldType,    \
           typename Enabled>                                                                    \
  struct __DssVisitField__<field_counter, F, WalkCtxType, DssFieldType, Enabled> {              \
    template<typename __DssFieldType>                                                           \
    using PartialF = F<field_counter, WalkCtxType, __DssFieldType>;                             \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                               \
      switch (field_ptr->field_case) {                                                          \
        OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_HOOK, type7field7case_tuple_seq)     \
        default:;                                                                               \
      }                                                                                         \
    }                                                                                           \
  };                                                                                            \
  template<template<int, class, class> class F, typename WalkCtxType, typename DssFieldType,    \
           typename Enabled>                                                                    \
  struct __DssVisitVerboseField__<field_counter, F, WalkCtxType, DssFieldType, Enabled> {       \
    template<typename __DssFieldType>                                                           \
    using PartialF = F<field_counter, WalkCtxType, __DssFieldType>;                             \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr, const char* __field_name__) {   \
      switch (field_ptr->field_case) {                                                          \
        OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_HOOK_VERBOSE,                        \
                             type7field7case_tuple_seq)                                         \
        default:;                                                                               \
      }                                                                                         \
    }                                                                                           \
  };                                                                                            \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType,              \
           typename DssFieldType, typename Enabled>                                             \
  struct __DssVisitStaticVerboseField__<field_counter, F, WalkCtxType, DssFieldType, Enabled> { \
    template<typename __DssFieldType>                                                           \
    using PartialF = F<__DssSelfType__, field_counter, WalkCtxType, __DssFieldType, true>;      \
    static void Call(WalkCtxType* ctx, const char* __oneof_name__) {                            \
      OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_HOOK_STATIC_VERBOSE,                   \
                           type7field7case_tuple_seq)                                           \
    }                                                                                           \
  };                                                                                            \
  template<template<int, class, class> class F, typename WalkCtxType, typename DssFieldType,    \
           typename Enabled>                                                                    \
  struct __DssVisitFieldUntil__<field_counter, F, WalkCtxType, DssFieldType, Enabled> {         \
    template<typename __DssFieldType>                                                           \
    using PartialF = F<field_counter, WalkCtxType, __DssFieldType>;                             \
    static bool Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                               \
      switch (field_ptr->field_case) {                                                          \
        OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_HOOK, type7field7case_tuple_seq)     \
        default:;                                                                               \
      }                                                                                         \
    }                                                                                           \
  };

#define _DSS_MAKE_UNION_FIELD_VISITOR_HOOK(field_type, field_name, field_case_value) \
  case field_case_value: {                                                           \
    return PartialF<field_type>::Call(ctx, &field_ptr->field_name);                  \
  }

#define _DSS_MAKE_UNION_FIELD_VISITOR_HOOK_VERBOSE(field_type, field_name, field_case_value) \
  case field_case_value: {                                                                   \
    const char* case_field_name = OF_PP_STRINGIZE(field_name);                               \
    return PartialF<field_type>::Call(ctx, &field_ptr->field_name, case_field_name);         \
  }

#define _DSS_MAKE_UNION_FIELD_VISITOR_HOOK_STATIC_VERBOSE(field_type, field_name, \
                                                          field_case_value)       \
  {                                                                               \
    const char* case_field_name = OF_PP_STRINGIZE(field_name);                    \
    PartialF<field_type>::Call(ctx, case_field_name, __oneof_name__);             \
  }

#define _DSS_BEGIN(field_counter, type)                                                       \
 private:                                                                                     \
  using __DssSelfType__ = type;                                                               \
                                                                                              \
 public:                                                                                      \
  template<int tpl_fld_counter, typename Enabled = void>                                      \
  struct __DssFieldType__;                                                                    \
  template<template<int, class, class> class F, typename WalkCtxType>                         \
  void __WalkField__(WalkCtxType* ctx) {                                                      \
    __DssFieldIter__<field_counter, F, WalkCtxType>::Call(ctx, this);                         \
  }                                                                                           \
  template<template<int, class, class> class F, typename WalkCtxType>                         \
  void __WalkVerboseField__(WalkCtxType* ctx) {                                               \
    __DssVerboseFieldIter__<field_counter, F, WalkCtxType>::Call(ctx, this);                  \
  }                                                                                           \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType>            \
  static void __WalkStaticVerboseField__(WalkCtxType* ctx) {                                  \
    __DssStaticVerboseFieldIter__<field_counter, F, WalkCtxType>::Call(ctx);                  \
  }                                                                                           \
  template<template<int, class, class> class F, typename WalkCtxType>                         \
  bool __WalkFieldUntil__(WalkCtxType* ctx) {                                                 \
    return __DssFieldIterUntil__<field_counter, F, WalkCtxType>::Call(ctx, this);             \
  }                                                                                           \
                                                                                              \
 private:                                                                                     \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename DssFieldType, typename Enabled = void>                                    \
  struct __DssVisitField__ {                                                                  \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                             \
      F<tpl_fld_counter, WalkCtxType, DssFieldType>::Call(ctx, field_ptr);                    \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename DssFieldType, typename Enabled = void>                                    \
  struct __DssVisitVerboseField__ {                                                           \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr, const char* __field_name__) { \
      F<tpl_fld_counter, WalkCtxType, DssFieldType>::Call(ctx, field_ptr, __field_name__);    \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<class, int, class, class, bool> class F,             \
           typename WalkCtxType, typename DssFieldType, typename Enabled = void>              \
  struct __DssVisitStaticVerboseField__ {                                                     \
    static void Call(WalkCtxType* ctx, const char* __field_name__) {                          \
      const char* __oneof_name__ = nullptr;                                                   \
      F<__DssSelfType__, tpl_fld_counter, WalkCtxType, DssFieldType, false>::Call(            \
          ctx, __field_name__, __oneof_name__);                                               \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename DssFieldType, typename Enabled = void>                                    \
  struct __DssVisitFieldUntil__ {                                                             \
    static bool Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                             \
      return F<tpl_fld_counter, WalkCtxType, DssFieldType>::Call(ctx, field_ptr);             \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename Enabled = void>                                                           \
  struct __DssFieldIter__ {                                                                   \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      __DssFieldIter__<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx, self);                 \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename Enabled = void>                                                           \
  struct __DssVerboseFieldIter__ {                                                            \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      __DssVerboseFieldIter__<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx, self);          \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<class, int, class, class, bool> class F,             \
           typename WalkCtxType, typename Enabled = void>                                     \
  struct __DssStaticVerboseFieldIter__ {                                                      \
    static void Call(WalkCtxType* ctx) {                                                      \
      __DssStaticVerboseFieldIter__<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx);          \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename Enabled = void>                                                           \
  struct __DssFieldIterUntil__ {                                                              \
    static bool Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      return __DssFieldIterUntil__<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx, self);     \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename Enabled = void>                                                           \
  struct __DssFieldReverseIter__ {                                                            \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      __DssFieldReverseIter__<tpl_fld_counter - 1, F, WalkCtxType>::Call(ctx, self);          \
    }                                                                                         \
  };                                                                                          \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled>       \
  struct __DssFieldReverseIter__<field_counter, F, WalkCtxType, Enabled> {                    \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {}                              \
  };                                                                                          \
  template<int tpl_fld_counter, typename Enabled = void>                                      \
  struct __DssFieldAlign4Counter__ {                                                          \
    static const int value = 1;                                                               \
  };                                                                                          \
  template<int tpl_fld_counter, typename Enabled = void>                                      \
  struct __DssFieldSize4Counter__ {                                                           \
    static const int value = 0;                                                               \
  };                                                                                          \
  template<int tpl_fld_counter, typename Enabled = void>                                      \
  struct __DssFieldOffsetOfFieldNumber__ {                                                    \
    constexpr static int Get() {                                                              \
      return __DssFieldOffsetOfFieldNumber__<tpl_fld_counter - 1, Enabled>::Get();            \
    }                                                                                         \
  };                                                                                          \
  template<typename Enabled>                                                                  \
  struct __DssFieldOffsetOfFieldNumber__<field_counter, Enabled> {                            \
    constexpr static int Get() { return 0; }                                                  \
  };                                                                                          \
  template<int tpl_fld_counter, typename Enabled = void>                                      \
  struct __DssStaticAssertFieldCounter__ {};                                                  \
                                                                                              \
  template<int tpl_fld_counter, typename Enabled = void>                                      \
  struct __DssAccumulatedAlignedSize4Counter__ {                                              \
    static const int value =                                                                  \
        ConstExprRoundUp<__DssAccumulatedAlignedSize4Counter__<tpl_fld_counter - 1>::value    \
                             + __DssFieldSize4Counter__<tpl_fld_counter - 1>::value,          \
                         __DssFieldAlign4Counter__<tpl_fld_counter>::value>();                \
  };                                                                                          \
  template<typename Enabled>                                                                  \
  struct __DssAccumulatedAlignedSize4Counter__<field_counter, Enabled> {                      \
    static const int value = 0;                                                               \
  };                                                                                          \
                                                                                              \
 public:                                                                                      \
  template<int field_index>                                                                   \
  struct __DssFieldOffset4FieldIndex__ {                                                      \
    static const int value = __DssAccumulatedAlignedSize4Counter__<field_index>::value;       \
  };

#define DSS_ASSERT_VERBOSE(dss_type)                                        \
  "\n\n\n    please check file " __FILE__ " (before line " OF_PP_STRINGIZE( \
      __LINE__) ") carefully\n"                                             \
                "    non " dss_type " member found before line " OF_PP_STRINGIZE(__LINE__) "\n\n"

#define _DSS_DEFINE_FIELD(field_counter, dss_type, field_type, field)                              \
 private:                                                                                          \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled>            \
  struct __DssFieldIter__<field_counter, F, WalkCtxType, Enabled> {                                \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      __DssVisitField__<field_counter, F, WalkCtxType, decltype(self->field)>::Call(ctx,           \
                                                                                    &self->field); \
      __DssFieldIter__<field_counter + 1, F, WalkCtxType>::Call(ctx, self);                        \
    }                                                                                              \
  };                                                                                               \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled>            \
  struct __DssVerboseFieldIter__<field_counter, F, WalkCtxType, Enabled> {                         \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      const char* __field_name__ = OF_PP_STRINGIZE(field);                                         \
      __DssVisitVerboseField__<field_counter, F, WalkCtxType, decltype(self->field)>::Call(        \
          ctx, &self->field, __field_name__);                                                      \
      __DssVerboseFieldIter__<field_counter + 1, F, WalkCtxType>::Call(ctx, self);                 \
    }                                                                                              \
  };                                                                                               \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType,                 \
           typename Enabled>                                                                       \
  struct __DssStaticVerboseFieldIter__<field_counter, F, WalkCtxType, Enabled> {                   \
    static void Call(WalkCtxType* ctx) {                                                           \
      const char* __field_name__ = OF_PP_STRINGIZE(field);                                         \
      __DssVisitStaticVerboseField__<                                                              \
          field_counter, F, WalkCtxType,                                                           \
          decltype(((__DssSelfType__*)nullptr)->field)>::Call(ctx, __field_name__);                \
      __DssStaticVerboseFieldIter__<field_counter + 1, F, WalkCtxType>::Call(ctx);                 \
    }                                                                                              \
  };                                                                                               \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled>            \
  struct __DssFieldIterUntil__<field_counter, F, WalkCtxType, Enabled> {                           \
    static bool Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      bool end =                                                                                   \
          __DssVisitFieldUntil__<field_counter, F, WalkCtxType, decltype(self->field)>::Call(      \
              ctx, &self->field);                                                                  \
      if (end) { return true; }                                                                    \
      return __DssFieldIterUntil__<field_counter + 1, F, WalkCtxType>::Call(ctx, self);            \
    }                                                                                              \
  };                                                                                               \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled>            \
  struct __DssFieldReverseIter__<field_counter, F, WalkCtxType, Enabled> {                         \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      __DssVisitField__<field_counter, F, WalkCtxType, decltype(self->field)>::Call(ctx,           \
                                                                                    &self->field); \
      __DssFieldReverseIter__<field_counter - 1, F, WalkCtxType>::Call(ctx, self);                 \
    }                                                                                              \
  };                                                                                               \
  template<typename Enabled>                                                                       \
  struct __DssFieldAlign4Counter__<field_counter, Enabled> {                                       \
    static const int value = alignof(field_type);                                                  \
  };                                                                                               \
  template<typename Enabled>                                                                       \
  struct __DssFieldSize4Counter__<field_counter, Enabled> {                                        \
    static const int value = sizeof(field_type);                                                   \
  };                                                                                               \
  template<typename Enabled>                                                                       \
  struct __DssFieldOffsetOfFieldNumber__<field_counter, Enabled> {                                 \
    constexpr static int Get() {                                                                   \
      static_assert(std::is_standard_layout<__DssSelfType__>::value, "");                          \
      return offsetof(__DssSelfType__, field);                                                     \
    }                                                                                              \
  };                                                                                               \
  template<typename Enabled>                                                                       \
  struct __DssStaticAssertFieldCounter__<field_counter, Enabled> {                                 \
    static void StaticAssert() {                                                                   \
      static const int kAccSize = __DssAccumulatedAlignedSize4Counter__<field_counter>::value;     \
      static_assert(kAccSize == __DssFieldOffsetOfFieldNumber__<field_counter>::Get(),             \
                    DSS_ASSERT_VERBOSE(dss_type));                                                 \
    }                                                                                              \
  };                                                                                               \
                                                                                                   \
 public:                                                                                           \
  template<typename Enabled>                                                                       \
  struct __DssFieldType__<field_counter, Enabled> {                                                \
    using type = field_type;                                                                       \
  };                                                                                               \
  [[maybe_unused]] static const int OF_PP_CAT(field, kDssFieldNumber) = field_counter;             \
  using OF_PP_CAT(field, DssFieldType) = field_type;                                               \
  [[maybe_unused]] static const int OF_PP_CAT(field, kDssFieldOffset) =                            \
      __DssAccumulatedAlignedSize4Counter__<field_counter>::value;

#define _DSS_END(field_counter, dss_type, type)                                         \
 public:                                                                                \
  template<template<int, class, class> class F, typename WalkCtxType>                   \
  void __ReverseWalkField__(WalkCtxType* ctx) {                                         \
    __DssFieldReverseIter__<field_counter, F, WalkCtxType>::Call(ctx, this);            \
  }                                                                                     \
                                                                                        \
 private:                                                                               \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled> \
  struct __DssFieldIter__<field_counter, F, WalkCtxType, Enabled> {                     \
    static void Call(WalkCtxType* ctx, type* self) {}                                   \
  };                                                                                    \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled> \
  struct __DssVerboseFieldIter__<field_counter, F, WalkCtxType, Enabled> {              \
    static void Call(WalkCtxType* ctx, type* self) {}                                   \
  };                                                                                    \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType,      \
           typename Enabled>                                                            \
  struct __DssStaticVerboseFieldIter__<field_counter, F, WalkCtxType, Enabled> {        \
    static void Call(WalkCtxType* ctx) {}                                               \
  };                                                                                    \
  template<template<int, class, class> class F, typename WalkCtxType, typename Enabled> \
  struct __DssFieldIterUntil__<field_counter, F, WalkCtxType, Enabled> {                \
    static bool Call(WalkCtxType* ctx, type* self) { return false; }                    \
  };                                                                                    \
  static void __DssStaticAssertStructSize__() {                                         \
    static const int kSize =                                                            \
        ConstExprRoundUp<__DssAccumulatedAlignedSize4Counter__<field_counter>::value,   \
                         alignof(type)>();                                              \
    static_assert((kSize == 0 && sizeof(type) == 1) || (kSize == sizeof(type)),         \
                  DSS_ASSERT_VERBOSE(dss_type));                                        \
  }

template<int x, int y>
constexpr int ConstExprRoundUp() {
  return (x + y - 1) / y * y;
}

template<bool is_pointer, typename Enabled = void>
struct GetterTrait {};

template<typename Enabled>
struct GetterTrait<false, Enabled> {
  template<typename T>
  static const T& Call(const T& data) {
    return data;
  }
};
template<typename Enabled>
struct GetterTrait<true, Enabled> {
  template<typename T>
  static const T& Call(const T* data) {
    return *data;
  }
};

template<bool is_pointer, typename Enabled = void>
struct MutableTrait {};

template<typename Enabled>
struct MutableTrait<false, Enabled> {
  template<typename T>
  static T* Call(T* data) {
    return data;
  }
};
template<typename Enabled>
struct MutableTrait<true, Enabled> {
  template<typename T>
  static T* Call(T** data) {
    return *data;
  }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_DSS_H_
