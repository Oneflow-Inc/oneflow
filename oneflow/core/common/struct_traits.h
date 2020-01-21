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
#define DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET() _DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET()
#define DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET(dss_type, size, offset) \
  _DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET(dss_type, size, offset)
#define DSS_STATIC_ASSERT_STRUCT_SIZE(dss_type, type) _DSS_STATIC_ASSERT_STRUCT_SIZE(dss_type, type)

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

#define _DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET()                                      \
 private:                                                                                   \
  template<int x, typename fake = void>                                                     \
  struct __DSS__FieldSizeOfCodeLine {                                                       \
    constexpr static int Get() { return 0; }                                                \
  };                                                                                        \
                                                                                            \
  template<int x, typename fake = void>                                                     \
  struct __DSS__OffsetOfDefiningField {                                                     \
    constexpr static int Get() { return __DSS__OffsetOfDefiningField<x - 1, fake>::Get(); } \
  };                                                                                        \
  template<typename fake>                                                                   \
  struct __DSS__OffsetOfDefiningField<__LINE__, fake> {                                     \
    constexpr static int Get() { return 0; }                                                \
  };                                                                                        \
                                                                                            \
  template<int x, typename fake = void>                                                     \
  struct __DSS__AccumulatedSizeOfDefinedFields {                                            \
    constexpr static int Get() {                                                            \
      return __DSS__AccumulatedSizeOfDefinedFields<x - 1, fake>::Get()                      \
             + __DSS__FieldSizeOfCodeLine<x - 1, fake>::Get();                              \
    }                                                                                       \
  };                                                                                        \
  template<typename fake>                                                                   \
  struct __DSS__AccumulatedSizeOfDefinedFields<__LINE__, fake> {                            \
    constexpr static int Get() { return 0; }                                                \
  };

#define ASSERT_VERBOSE(dss_type)                                            \
  "\n\n\n    please check file " __FILE__ " (before line " OF_PP_STRINGIZE( \
      __LINE__) ") carefully\n"                                             \
                "    non " dss_type " member found before line " OF_PP_STRINGIZE(__LINE__) "\n\n"

#define _DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET(dss_type, size, offset)                \
 private:                                                                                 \
  template<typename fake>                                                                 \
  struct __DSS__FieldSizeOfCodeLine<__LINE__, fake> {                                     \
    constexpr static int Get() { return size; }                                           \
  };                                                                                      \
  template<typename fake>                                                                 \
  struct __DSS__OffsetOfDefiningField<__LINE__, fake> {                                   \
    constexpr static int Get() { return (int)(long long)(offset); }                       \
  };                                                                                      \
  static void OF_PP_CAT(__DSS__StaticAssertFieldCodeLineFieldSizeAndOffset, __LINE__)() { \
    static_assert(__DSS__AccumulatedSizeOfDefinedFields<__LINE__>::Get()                  \
                      == __DSS__OffsetOfDefiningField<__LINE__>::Get(),                   \
                  ASSERT_VERBOSE(dss_type));                                              \
  }

#define _DSS_STATIC_ASSERT_STRUCT_SIZE(dss_type, type)                                    \
 private:                                                                                 \
  static void __DSS__StaticAssertStructSize() {                                           \
    static_assert(__DSS__AccumulatedSizeOfDefinedFields<__LINE__>::Get() == sizeof(type), \
                  ASSERT_VERBOSE(dss_type));                                              \
  }
}

#endif  // ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
