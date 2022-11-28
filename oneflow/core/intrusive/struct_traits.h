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
#ifndef ONEFLOW_CORE_INTRUSIVE_STRUCT_MACRO_TRAITS_H_
#define ONEFLOW_CORE_INTRUSIVE_STRUCT_MACRO_TRAITS_H_

#include <cstddef>
#include <type_traits>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {
namespace intrusive {

template<typename T, typename F, F T::*ptr2member>
struct PtrStructField {
  using struct_type = T;
  using field_type = F;

  static T* StructPtr4FieldPtr(const F* field_ptr) {
    int offset_value = reinterpret_cast<long long>(&(((T*)nullptr)->*ptr2member));
    return (T*)((const_cast<char*>(reinterpret_cast<const char*>(field_ptr))) - offset_value);
  }
  static F* FieldPtr4StructPtr(const T* struct_ptr) {
    return &(const_cast<T*>(struct_ptr)->*ptr2member);
  }
};

template<typename T, typename F, int offset>
struct OffsetStructField {
  using struct_type = T;
  using field_type = F;
  static const int offset_value = offset;

  static T* StructPtr4FieldPtr(const F* field_ptr) {
    return (T*)((const_cast<char*>(reinterpret_cast<const char*>(field_ptr))) - offset_value);
  }
  static F* FieldPtr4StructPtr(const T* struct_ptr) {
    return (F*)((const_cast<char*>(reinterpret_cast<const char*>(struct_ptr))) + offset_value);
  }
};

#define INTRUSIVE_FIELD(struct_type, field_name)                                        \
  intrusive::PtrStructField<struct_type, decltype(((struct_type*)nullptr)->field_name), \
                            &struct_type::field_name>

template<typename X, typename Y>
struct ComposeStructField {
  static_assert(std::is_same<typename X::field_type, typename Y::struct_type>::value,
                "invalid type");
  using struct_type = typename X::struct_type;
  using field_type = typename Y::field_type;
  static struct_type* StructPtr4FieldPtr(const field_type* field_ptr) {
    return X::StructPtr4FieldPtr(Y::StructPtr4FieldPtr(field_ptr));
  }
  static field_type* FieldPtr4StructPtr(const struct_type* struct_ptr) {
    return Y::FieldPtr4StructPtr(X::FieldPtr4StructPtr(struct_ptr));
  }
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

}  // namespace intrusive
}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_STRUCT_MACRO_TRAITS_H_
