#ifndef ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
#define ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_

namespace oneflow {

#define STRUCT_FIELD(T, field) StructField<T, STRUCT_FIELD_OFFSET(T, field)>
#define DEFINE_STRUCT_FIELD(T, field)                        \
  template<>                                                 \
  struct StructField<T, STRUCT_FIELD_OFFSET(T, field)> final \
      : public StructFieldImpl<T, STRUCT_FIELD_TYPE(T, field), STRUCT_FIELD_OFFSET(T, field)> {};

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

#define STRUCT_FIELD_TYPE(T, field) decltype(((T*)nullptr)->field)
#define STRUCT_FIELD_OFFSET(T, field) ((int)(long long)&((T*)nullptr)->field)
}

#endif  // ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
