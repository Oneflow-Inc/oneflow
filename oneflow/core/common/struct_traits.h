#ifndef ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
#define ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_

#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define STRUCT_FIELD(T, field) StructField<T, STRUCT_FIELD_OFFSET(T, field)>
#define DEFINE_STRUCT_FIELD(T, field)                        \
  template<>                                                 \
  struct StructField<T, STRUCT_FIELD_OFFSET(T, field)> final \
      : public StructFieldImpl<T, STRUCT_FIELD_TYPE(T, field), STRUCT_FIELD_OFFSET(T, field)> {};
#define STRUCT_FIELD_TYPE(T, field) decltype(((T*)nullptr)->field)
#define STRUCT_FIELD_OFFSET(T, field) ((int)(long long)&((T*)nullptr)->field)

#define DEFINE_GETTER(field_type, field_name) _DEFINE_GETTER(field_type, field_name)

#define DEFINE_MUTABLE(field_type, field_name) _DEFINE_MUTABLE(field_type, field_name)

#define DEFINE_SETTER(is_scalar, field_type, field_name) \
  _DEFINE_SETTER(is_scalar, field_type, field_name)

// DSS is short for domain specific struct
#define BEGIN_DSS(base_byte_size) _BEGIN_DSS(__COUNTER__, base_byte_size)
#define DSS_DEFINE_FIELD(dss_type, type, field) \
  _DSS_DEFINE_FIELD(__COUNTER__, dss_type, type, field)
#define END_DSS(dss_type, type) _END_DSS(__COUNTER__, dss_type, type)

// details
template<typename T, int offset>
struct StructField {};

template<typename T, typename F, int offset>
struct StructFieldImpl {
  using struct_type = T;
  using field_type = F;
  static const int offset_value = offset;

  static T* StructPtr4FieldPtr(const F* field_ptr) {
    return (T*)(((char*)field_ptr) - offset_value);
  }
  static F* FieldPtr4StructPtr(const T* struct_ptr) {
    return (F*)(((char*)struct_ptr) + offset_value);
  }
};

template<int x, int y>
constexpr int ConstExprRoundUp() {
  return (x + y - 1) / y * y;
}

#define _BEGIN_DSS(define_counter, base_byte_size)                                              \
 private:                                                                                       \
  template<int counter, typename fake = void>                                                   \
  struct __DSS__FieldAlign4Counter {                                                            \
    constexpr static int Get() { return 1; }                                                    \
  };                                                                                            \
                                                                                                \
  template<int counter, typename fake = void>                                                   \
  struct __DSS__FieldSize4Counter {                                                             \
    constexpr static int Get() { return base_byte_size; }                                       \
  };                                                                                            \
                                                                                                \
  template<int counter, typename fake = void>                                                   \
  struct __DSS__FieldOffset4Counter {                                                           \
    constexpr static int Get() { return __DSS__FieldOffset4Counter<counter - 1, fake>::Get(); } \
  };                                                                                            \
  template<typename fake>                                                                       \
  struct __DSS__FieldOffset4Counter<define_counter, fake> {                                     \
    constexpr static int Get() { return base_byte_size; }                                       \
  };                                                                                            \
                                                                                                \
  template<int counter, typename fake = void>                                                   \
  struct __DSS__AccumulatedAlignedSize4Counter {                                                \
    constexpr static int Get() {                                                                \
      return ConstExprRoundUp<__DSS__AccumulatedAlignedSize4Counter<counter - 1, fake>::Get()   \
                                  + __DSS__FieldSize4Counter<counter - 1, fake>::Get(),         \
                              __DSS__FieldAlign4Counter<counter, fake>::Get()>();               \
    }                                                                                           \
  };                                                                                            \
  template<typename fake>                                                                       \
  struct __DSS__AccumulatedAlignedSize4Counter<define_counter, fake> {                          \
    constexpr static int Get() { return 0; }                                                    \
  };

#define ASSERT_VERBOSE(dss_type)                                            \
  "\n\n\n    please check file " __FILE__ " (before line " OF_PP_STRINGIZE( \
      __LINE__) ") carefully\n"                                             \
                "    non " dss_type " member found before line " OF_PP_STRINGIZE(__LINE__) "\n\n"

#define _DSS_DEFINE_FIELD(define_counter, dss_type, type, field)                              \
 private:                                                                                     \
  template<typename fake>                                                                     \
  struct __DSS__FieldAlign4Counter<define_counter, fake> {                                    \
    constexpr static int Get() { return alignof(((type*)nullptr)->field); }                   \
  };                                                                                          \
  template<typename fake>                                                                     \
  struct __DSS__FieldSize4Counter<define_counter, fake> {                                     \
    constexpr static int Get() { return sizeof(((type*)nullptr)->field); }                    \
  };                                                                                          \
  template<typename fake>                                                                     \
  struct __DSS__FieldOffset4Counter<define_counter, fake> {                                   \
    constexpr static int Get() { return offsetof(type, field); }                              \
  };                                                                                          \
  static void OF_PP_CAT(__DSS__StaticAssertFieldCounter, define_counter)() {                  \
    static const int kAccSize = __DSS__AccumulatedAlignedSize4Counter<define_counter>::Get(); \
    static_assert(kAccSize == __DSS__FieldOffset4Counter<define_counter>::Get(),              \
                  ASSERT_VERBOSE(dss_type));                                                  \
  }

#define _END_DSS(counter, dss_type, type)                                                         \
 private:                                                                                         \
  static void __DSS__StaticAssertStructSize() {                                                   \
    static const int kSize =                                                                      \
        ConstExprRoundUp<__DSS__AccumulatedAlignedSize4Counter<counter>::Get(), alignof(type)>(); \
    static_assert((kSize == 0 && sizeof(type) == 1) || (kSize == sizeof(type)),                   \
                  ASSERT_VERBOSE(dss_type));                                                      \
  }
}

template<bool is_ptr, typename Enabled = void>
struct GetterTrait {};

template<typename Enabled>
struct GetterTrait<false, Enabled> {
  template<typename T>
  static const T& Apply(const T& data) {
    return data;
  }
};
template<typename Enabled>
struct GetterTrait<true, Enabled> {
  template<typename T>
  static const T& Apply(const T* data) {
    return *data;
  }
};

template<bool is_ptr, typename Enabled = void>
struct MutableTrait {};

template<typename Enabled>
struct MutableTrait<false, Enabled> {
  template<typename T>
  static T* Apply(T* data) {
    return data;
  }
};
template<typename Enabled>
struct MutableTrait<true, Enabled> {
  template<typename T>
  static T* Apply(T** data) {
    return *data;
  }
};

#define _DEFINE_GETTER(field_type, field_name)                                              \
  const typename std::conditional<std::is_pointer<field_type>::value,                       \
                                  std::remove_pointer<field_type>::type, field_type>::type& \
  field_name() const {                                                                      \
    return GetterTrait<std::is_pointer<field_type>::value>::Apply(this->field_name##_);     \
  }

#define _DEFINE_MUTABLE(field_type, field_name)                                           \
  typename std::conditional<std::is_pointer<field_type>::value,                           \
                            std::remove_pointer<field_type>::type, field_type>::type*     \
      mutable_##field_name() {                                                            \
    return MutableTrait<std::is_pointer<field_type>::value>::Apply(&this->field_name##_); \
  }

#define _DEFINE_SETTER(is_scalar, field_type, field_name)                          \
  template<typename T>                                                             \
  void set_##field_name(const T& val) {                                            \
    static_assert(is_scalar<T>::value, "setter doesn't support non-scalar field"); \
    *this->mutable_##field_name() = val;                                           \
  }

#endif  // ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
