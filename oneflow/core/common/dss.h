#ifndef ONEFLOW_CORE_COMMON_DSS_H_
#define ONEFLOW_CORE_COMMON_DSS_H_

#include <cstddef>
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/struct_traits.h"

namespace oneflow {

// DSS is short for domain specific struct
#define BEGIN_DSS(field_counter, type, base_byte_size) \
  _BEGIN_DSS(field_counter, type, base_byte_size)
#define DSS_DEFINE_FIELD(field_counter, dss_type, field) \
  _DSS_DEFINE_FIELD(field_counter, dss_type, field)
#define END_DSS(field_counter, dss_type, type) _END_DSS(field_counter, dss_type, type)
#define DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, field_case, type7field7case_tuple_seq) \
  _DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, field_case, type7field7case_tuple_seq)
#define DSS_GET_FIELD_COUNTER() __COUNTER__

// details

#define _DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, field_case, type7field7case_tuple_seq) \
  template<template<int, class, class> class F, typename WalkCtxType, typename DssFieldType,  \
           typename fake>                                                                     \
  struct __DSS__VisitField<field_counter, F, WalkCtxType, DssFieldType, fake> {               \
    template<typename __DssFieldType>                                                         \
    using PartialF = F<field_counter, WalkCtxType, __DssFieldType>;                           \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                             \
      switch (field_ptr->field_case) {                                                        \
        OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_ENTRY, type7field7case_tuple_seq)  \
        default:;                                                                             \
      }                                                                                       \
    }                                                                                         \
  };                                                                                          \
  template<template<int, class, class> class F, typename WalkCtxType, typename DssFieldType,  \
           typename fake>                                                                     \
  struct __DSS__VisitVerboseField<field_counter, F, WalkCtxType, DssFieldType, fake> {        \
    template<typename __DssFieldType>                                                         \
    using PartialF = F<field_counter, WalkCtxType, __DssFieldType>;                           \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr, const char* __field_name__) { \
      switch (field_ptr->field_case) {                                                        \
        OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_ENTRY_VERBOSE,                     \
                             type7field7case_tuple_seq)                                       \
        default:;                                                                             \
      }                                                                                       \
    }                                                                                         \
  };                                                                                          \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType,            \
           typename DssFieldType, typename fake>                                              \
  struct __DSS__VisitStaticVerboseField<field_counter, F, WalkCtxType, DssFieldType, fake> {  \
    template<typename __DssFieldType>                                                         \
    using PartialF = F<__DssSelfType__, field_counter, WalkCtxType, __DssFieldType, true>;    \
    static void Call(WalkCtxType* ctx, const char* __oneof_name__) {                          \
      OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_ENTRY_STATIC_VERBOSE,                \
                           type7field7case_tuple_seq)                                         \
    }                                                                                         \
  };                                                                                          \
  template<template<int, class, class> class F, typename WalkCtxType, typename DssFieldType,  \
           typename fake>                                                                     \
  struct __DSS__VisitFieldUntil<field_counter, F, WalkCtxType, DssFieldType, fake> {          \
    template<typename __DssFieldType>                                                         \
    using PartialF = F<field_counter, WalkCtxType, __DssFieldType>;                           \
    static bool Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                             \
      switch (field_ptr->field_case) {                                                        \
        OF_PP_FOR_EACH_TUPLE(_DSS_MAKE_UNION_FIELD_VISITOR_ENTRY, type7field7case_tuple_seq)  \
        default:;                                                                             \
      }                                                                                       \
    }                                                                                         \
  };

#define _DSS_MAKE_UNION_FIELD_VISITOR_ENTRY(field_type, field_name, field_case_value) \
  case field_case_value: {                                                            \
    return PartialF<field_type>::Call(ctx, &field_ptr->field_name);                   \
  }

#define _DSS_MAKE_UNION_FIELD_VISITOR_ENTRY_VERBOSE(field_type, field_name, field_case_value) \
  case field_case_value: {                                                                    \
    const char* case_field_name = OF_PP_STRINGIZE(field_name);                                \
    return PartialF<field_type>::Call(ctx, &field_ptr->field_name, case_field_name);          \
  }

#define _DSS_MAKE_UNION_FIELD_VISITOR_ENTRY_STATIC_VERBOSE(field_type, field_name, \
                                                           field_case_value)       \
  {                                                                                \
    const char* case_field_name = OF_PP_STRINGIZE(field_name);                     \
    PartialF<field_type>::Call(ctx, case_field_name, __oneof_name__);              \
  }

#define _BEGIN_DSS(field_counter, type, base_byte_size)                                       \
 private:                                                                                     \
  using __DssSelfType__ = type;                                                               \
                                                                                              \
 public:                                                                                      \
  template<template<int, class, class> class F, typename WalkCtxType>                         \
  void __WalkField__(WalkCtxType* ctx) {                                                      \
    __DSS__FieldIter<field_counter, F, WalkCtxType>::Call(ctx, this);                         \
  }                                                                                           \
  template<template<int, class, class> class F, typename WalkCtxType>                         \
  void __WalkVerboseField__(WalkCtxType* ctx) {                                               \
    __DSS__VerboseFieldIter<field_counter, F, WalkCtxType>::Call(ctx, this);                  \
  }                                                                                           \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType>            \
  static void __WalkStaticVerboseField__(WalkCtxType* ctx) {                                  \
    __DSS__StaticVerboseFieldIter<field_counter, F, WalkCtxType>::Call(ctx);                  \
  }                                                                                           \
  template<template<int, class, class> class F, typename WalkCtxType>                         \
  bool __WalkFieldUntil__(WalkCtxType* ctx) {                                                 \
    return __DSS__FieldIterUntil<field_counter, F, WalkCtxType>::Call(ctx, this);             \
  }                                                                                           \
  template<int tpl_field_counter, typename FieldType, typename Enable = void>                 \
  struct __DssFieldTypeId__ final {                                                           \
    static std::string Call() { return typeid(FieldType).name(); }                            \
  };                                                                                          \
                                                                                              \
 private:                                                                                     \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename DssFieldType, typename fake = void>                                       \
  struct __DSS__VisitField {                                                                  \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                             \
      F<tpl_fld_counter, WalkCtxType, DssFieldType>::Call(ctx, field_ptr);                    \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename DssFieldType, typename fake = void>                                       \
  struct __DSS__VisitVerboseField {                                                           \
    static void Call(WalkCtxType* ctx, DssFieldType* field_ptr, const char* __field_name__) { \
      F<tpl_fld_counter, WalkCtxType, DssFieldType>::Call(ctx, field_ptr, __field_name__);    \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<class, int, class, class, bool> class F,             \
           typename WalkCtxType, typename DssFieldType, typename fake = void>                 \
  struct __DSS__VisitStaticVerboseField {                                                     \
    static void Call(WalkCtxType* ctx, const char* __field_name__) {                          \
      const char* __oneof_name__ = nullptr;                                                   \
      F<__DssSelfType__, tpl_fld_counter, WalkCtxType, DssFieldType, false>::Call(            \
          ctx, __field_name__, __oneof_name__);                                               \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename DssFieldType, typename fake = void>                                       \
  struct __DSS__VisitFieldUntil {                                                             \
    static bool Call(WalkCtxType* ctx, DssFieldType* field_ptr) {                             \
      return F<tpl_fld_counter, WalkCtxType, DssFieldType>::Call(ctx, field_ptr);             \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename fake = void>                                                              \
  struct __DSS__FieldIter {                                                                   \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      __DSS__FieldIter<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx, self);                 \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename fake = void>                                                              \
  struct __DSS__VerboseFieldIter {                                                            \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      __DSS__VerboseFieldIter<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx, self);          \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<class, int, class, class, bool> class F,             \
           typename WalkCtxType, typename fake = void>                                        \
  struct __DSS__StaticVerboseFieldIter {                                                      \
    static void Call(WalkCtxType* ctx) {                                                      \
      __DSS__StaticVerboseFieldIter<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx);          \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename fake = void>                                                              \
  struct __DSS__FieldIterUntil {                                                              \
    static bool Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      return __DSS__FieldIterUntil<tpl_fld_counter + 1, F, WalkCtxType>::Call(ctx, self);     \
    }                                                                                         \
  };                                                                                          \
  template<int tpl_fld_counter, template<int, class, class> class F, typename WalkCtxType,    \
           typename fake = void>                                                              \
  struct __DSS__FieldReverseIter {                                                            \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                               \
      __DSS__FieldReverseIter<tpl_fld_counter - 1, F, WalkCtxType>::Call(ctx, self);          \
    }                                                                                         \
  };                                                                                          \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>          \
  struct __DSS__FieldReverseIter<field_counter, F, WalkCtxType, fake> {                       \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {}                              \
  };                                                                                          \
  template<int tpl_fld_counter, typename fake = void>                                         \
  struct __DSS__FieldAlign4Counter {                                                          \
    constexpr static int Get() { return 1; }                                                  \
  };                                                                                          \
                                                                                              \
  template<int tpl_fld_counter, typename fake = void>                                         \
  struct __DSS__FieldSize4Counter {                                                           \
    constexpr static int Get() { return base_byte_size; }                                     \
  };                                                                                          \
                                                                                              \
  template<int tpl_fld_counter, typename fake = void>                                         \
  struct __DSS__FieldOffset4Counter {                                                         \
    constexpr static int Get() {                                                              \
      return __DSS__FieldOffset4Counter<tpl_fld_counter - 1, fake>::Get();                    \
    }                                                                                         \
  };                                                                                          \
  template<typename fake>                                                                     \
  struct __DSS__FieldOffset4Counter<field_counter, fake> {                                    \
    constexpr static int Get() { return base_byte_size; }                                     \
  };                                                                                          \
  template<int tpl_fld_counter, typename fake = void>                                         \
  struct __DSS__StaticAssertFieldCounter {};                                                  \
                                                                                              \
  template<int tpl_fld_counter, typename fake = void>                                         \
  struct __DSS__AccumulatedAlignedSize4Counter {                                              \
    constexpr static int Get() {                                                              \
      return ConstExprRoundUp<                                                                \
          __DSS__AccumulatedAlignedSize4Counter<tpl_fld_counter - 1, fake>::Get()             \
              + __DSS__FieldSize4Counter<tpl_fld_counter - 1, fake>::Get(),                   \
          __DSS__FieldAlign4Counter<tpl_fld_counter, fake>::Get()>();                         \
    }                                                                                         \
  };                                                                                          \
  template<typename fake>                                                                     \
  struct __DSS__AccumulatedAlignedSize4Counter<field_counter, fake> {                         \
    constexpr static int Get() { return 0; }                                                  \
  };

#define DSS_ASSERT_VERBOSE(dss_type)                                        \
  "\n\n\n    please check file " __FILE__ " (before line " OF_PP_STRINGIZE( \
      __LINE__) ") carefully\n"                                             \
                "    non " dss_type " member found before line " OF_PP_STRINGIZE(__LINE__) "\n\n"

#define _DSS_DEFINE_FIELD(field_counter, dss_type, field)                                          \
 public:                                                                                           \
  constexpr static int OF_PP_CAT(field, DssFieldOffset)() {                                        \
    static_assert(std::is_standard_layout<__DssSelfType__>::value, "");                            \
    return offsetof(__DssSelfType__, field);                                                       \
  }                                                                                                \
                                                                                                   \
 private:                                                                                          \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>               \
  struct __DSS__FieldIter<field_counter, F, WalkCtxType, fake> {                                   \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      __DSS__VisitField<field_counter, F, WalkCtxType, decltype(self->field)>::Call(ctx,           \
                                                                                    &self->field); \
      __DSS__FieldIter<field_counter + 1, F, WalkCtxType>::Call(ctx, self);                        \
    }                                                                                              \
  };                                                                                               \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>               \
  struct __DSS__VerboseFieldIter<field_counter, F, WalkCtxType, fake> {                            \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      const char* __field_name__ = OF_PP_STRINGIZE(field);                                         \
      __DSS__VisitVerboseField<field_counter, F, WalkCtxType, decltype(self->field)>::Call(        \
          ctx, &self->field, __field_name__);                                                      \
      __DSS__VerboseFieldIter<field_counter + 1, F, WalkCtxType>::Call(ctx, self);                 \
    }                                                                                              \
  };                                                                                               \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType, typename fake>  \
  struct __DSS__StaticVerboseFieldIter<field_counter, F, WalkCtxType, fake> {                      \
    static void Call(WalkCtxType* ctx) {                                                           \
      const char* __field_name__ = OF_PP_STRINGIZE(field);                                         \
      __DSS__VisitStaticVerboseField<                                                              \
          field_counter, F, WalkCtxType,                                                           \
          decltype(((__DssSelfType__*)nullptr)->field)>::Call(ctx, __field_name__);                \
      __DSS__StaticVerboseFieldIter<field_counter + 1, F, WalkCtxType>::Call(ctx);                 \
    }                                                                                              \
  };                                                                                               \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>               \
  struct __DSS__FieldIterUntil<field_counter, F, WalkCtxType, fake> {                              \
    static bool Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      bool end =                                                                                   \
          __DSS__VisitFieldUntil<field_counter, F, WalkCtxType, decltype(self->field)>::Call(      \
              ctx, &self->field);                                                                  \
      if (end) { return true; }                                                                    \
      return __DSS__FieldIterUntil<field_counter + 1, F, WalkCtxType>::Call(ctx, self);            \
    }                                                                                              \
  };                                                                                               \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>               \
  struct __DSS__FieldReverseIter<field_counter, F, WalkCtxType, fake> {                            \
    static void Call(WalkCtxType* ctx, __DssSelfType__* self) {                                    \
      __DSS__VisitField<field_counter, F, WalkCtxType, decltype(self->field)>::Call(ctx,           \
                                                                                    &self->field); \
      __DSS__FieldReverseIter<field_counter - 1, F, WalkCtxType>::Call(ctx, self);                 \
    }                                                                                              \
  };                                                                                               \
  template<typename fake>                                                                          \
  struct __DSS__FieldAlign4Counter<field_counter, fake> {                                          \
    constexpr static int Get() { return alignof(((__DssSelfType__*)nullptr)->field); }             \
  };                                                                                               \
  template<typename fake>                                                                          \
  struct __DSS__FieldSize4Counter<field_counter, fake> {                                           \
    constexpr static int Get() { return sizeof(((__DssSelfType__*)nullptr)->field); }              \
  };                                                                                               \
  template<typename fake>                                                                          \
  struct __DSS__FieldOffset4Counter<field_counter, fake> {                                         \
    constexpr static int Get() {                                                                   \
      static_assert(std::is_standard_layout<__DssSelfType__>::value, "");                          \
      return offsetof(__DssSelfType__, field);                                                     \
    }                                                                                              \
  };                                                                                               \
  template<typename fake>                                                                          \
  struct __DSS__StaticAssertFieldCounter<field_counter, fake> {                                    \
    static void StaticAssert() {                                                                   \
      static const int kAccSize = __DSS__AccumulatedAlignedSize4Counter<field_counter>::Get();     \
      static_assert(kAccSize == __DSS__FieldOffset4Counter<field_counter>::Get(),                  \
                    DSS_ASSERT_VERBOSE(dss_type));                                                 \
    }                                                                                              \
  };

#define _END_DSS(field_counter, dss_type, type)                                                   \
 public:                                                                                          \
  template<template<int, class, class> class F, typename WalkCtxType>                             \
  void __ReverseWalkField__(WalkCtxType* ctx) {                                                   \
    __DSS__FieldReverseIter<field_counter, F, WalkCtxType>::Call(ctx, this);                      \
  }                                                                                               \
                                                                                                  \
 private:                                                                                         \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>              \
  struct __DSS__FieldIter<field_counter, F, WalkCtxType, fake> {                                  \
    static void Call(WalkCtxType* ctx, type* self) {}                                             \
  };                                                                                              \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>              \
  struct __DSS__VerboseFieldIter<field_counter, F, WalkCtxType, fake> {                           \
    static void Call(WalkCtxType* ctx, type* self) {}                                             \
  };                                                                                              \
  template<template<class, int, class, class, bool> class F, typename WalkCtxType, typename fake> \
  struct __DSS__StaticVerboseFieldIter<field_counter, F, WalkCtxType, fake> {                     \
    static void Call(WalkCtxType* ctx) {}                                                         \
  };                                                                                              \
  template<template<int, class, class> class F, typename WalkCtxType, typename fake>              \
  struct __DSS__FieldIterUntil<field_counter, F, WalkCtxType, fake> {                             \
    static bool Call(WalkCtxType* ctx, type* self) { return false; }                              \
  };                                                                                              \
  static void __DSS__StaticAssertStructSize() {                                                   \
    static const int kSize =                                                                      \
        ConstExprRoundUp<__DSS__AccumulatedAlignedSize4Counter<field_counter>::Get(),             \
                         alignof(type)>();                                                        \
    static_assert((kSize == 0 && sizeof(type) == 1) || (kSize == sizeof(type)),                   \
                  DSS_ASSERT_VERBOSE(dss_type));                                                  \
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

#endif  // ONEFLOW_CORE_COMMON_DSS_H_
