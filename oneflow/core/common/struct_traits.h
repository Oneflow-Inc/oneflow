#ifndef ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
#define ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_

#include <cstddef>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define STRUCT_FIELD(T, field) \
  StructField<T, STRUCT_FIELD_TYPE(T, field), STRUCT_FIELD_OFFSET(T, field)>
#define STRUCT_FIELD_TYPE(T, field) decltype(((T*)nullptr)->field)
#define STRUCT_FIELD_OFFSET(T, field) offsetof(T, field)

// details
template<typename T, typename F, int offset>
struct StructField {
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
}

#endif  // ONEFLOW_CORE_COMMON_STRUCT_MACRO_TRAITS_H_
