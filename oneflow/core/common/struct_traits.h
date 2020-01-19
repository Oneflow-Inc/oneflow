#ifndef ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
#define ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_

namespace oneflow {

#define STRUCT_FIELD(T, field) StructField<T, STRUCT_FIELD_OFFSET(T, field)>
#define DEFINE_STRUCT_FIELD(T, field)                        \
  template<>                                                 \
  struct StructField<T, STRUCT_FIELD_OFFSET(T, field)> final \
      : public StructFieldImpl<T, STRUCT_FIELD_TYPE(T, field), STRUCT_FIELD_OFFSET(T, field)> {};
#define STRUCT_FIELD_TYPE(T, field) decltype(((T*)nullptr)->field)
#define STRUCT_FIELD_OFFSET(T, field) ((int)(long long)&((T*)nullptr)->field)

// DSS is short for domain specific struct
#define DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET() \
  _DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET(__COUNTER__)
#define DSS_DEFINE_AND_CHECK_CODE_LINE_FIELD(dss_type, type, field) \
  _DSS_DEFINE_AND_CHECK_CODE_LINE_FIELD(__COUNTER__, dss_type, type, field)
#define DSS_STATIC_ASSERT_STRUCT_SIZE(dss_type, type) \
  _DSS_STATIC_ASSERT_STRUCT_SIZE(__COUNTER__, dss_type, type)

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

#define _DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET(define_counter)                            \
 private:                                                                                       \
  template<int counter, typename fake = void>                                                   \
  struct __DSS__FieldSize4Counter {                                                             \
    constexpr static int Get() { return 0; }                                                    \
  };                                                                                            \
                                                                                                \
  template<int counter, typename fake = void>                                                   \
  struct __DSS__FieldOffset4Counter {                                                           \
    constexpr static int Get() { return __DSS__FieldOffset4Counter<counter - 1, fake>::Get(); } \
  };                                                                                            \
  template<typename fake>                                                                       \
  struct __DSS__FieldOffset4Counter<define_counter, fake> {                                     \
    constexpr static int Get() { return 0; }                                                    \
  };                                                                                            \
                                                                                                \
  template<int counter, typename fake = void>                                                   \
  struct __DSS__AccumulatedAlignedSize4Counter {                                                \
    constexpr static int Get() {                                                                \
      return __DSS__AccumulatedAlignedSize4Counter<counter - 1, fake>::Get()                    \
             + __DSS__FieldSize4Counter<counter - 1, fake>::Get();                              \
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

#define _DSS_DEFINE_AND_CHECK_CODE_LINE_FIELD(define_counter, dss_type, type, field) \
 private:                                                                            \
  template<typename fake>                                                            \
  struct __DSS__FieldSize4Counter<define_counter, fake> {                            \
    constexpr static int Get() { return sizeof(((type*)nullptr)->field); }           \
  };                                                                                 \
  template<typename fake>                                                            \
  struct __DSS__FieldOffset4Counter<define_counter, fake> {                          \
    constexpr static int Get() { return STRUCT_FIELD_OFFSET(type, field); }          \
  };                                                                                 \
  static void OF_PP_CAT(__DSS__StaticAssertFieldCounter, define_counter)() {         \
    static_assert(__DSS__AccumulatedAlignedSize4Counter<define_counter>::Get()       \
                      == __DSS__FieldOffset4Counter<define_counter>::Get(),          \
                  ASSERT_VERBOSE(dss_type));                                         \
  }

#define _DSS_STATIC_ASSERT_STRUCT_SIZE(define_counter, dss_type, type)                          \
 private:                                                                                       \
  static void __DSS__StaticAssertStructSize() {                                                 \
    static_assert(__DSS__AccumulatedAlignedSize4Counter<define_counter>::Get() == sizeof(type), \
                  ASSERT_VERBOSE(dss_type));                                                    \
  }
}

#endif  // ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
