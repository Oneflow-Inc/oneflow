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
#ifndef ONEFLOW_CORE_OBJECT_MSG_STRUCT_MACRO_TRAITS_H_
#define ONEFLOW_CORE_OBJECT_MSG_STRUCT_MACRO_TRAITS_H_

#include <cstddef>
#include <type_traits>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define STRUCT_FIELD(T, field) \
  StructField<T, STRUCT_FIELD_TYPE(T, field), STRUCT_FIELD_OFFSET(T, field)>
#define STRUCT_FIELD_TYPE(T, field) decltype(((T*)nullptr)->field)
#define STRUCT_FIELD_OFFSET(T, field) offsetof(T, field)

#define OF_PUBLIC public:
#define OF_PRIVATE private:

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

template<typename X, typename Y>
struct ComposeStructField {
  static_assert(std::is_same<typename X::field_type, typename Y::struct_type>::value,
                "invalid type");
  using type = StructField<typename X::struct_type, typename Y::field_type,
                           X::offset_value + Y::offset_value>;
};

template<typename T>
struct ConstStruct {
  using type = const T;
};
template<typename T>
struct ConstStruct<const T> {
  using type = const T;
};

template<typename T>
using ConstType = typename ConstStruct<T>::type;

template<typename T>
struct ConstRefOrPtrStruct {
  using type = ConstType<T>&;
};

template<typename T>
struct ConstRefOrPtrStruct<T*> {
  using type = ConstType<T>*;
};

template<typename T>
using ConstRefOrPtr = typename ConstRefOrPtrStruct<T>::type;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_STRUCT_MACRO_TRAITS_H_
