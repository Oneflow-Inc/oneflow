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
#ifndef ONEFLOW_CORE_INTRUSIVE_FOR_EACH_H_
#define ONEFLOW_CORE_INTRUSIVE_FOR_EACH_H_

#include "oneflow/core/intrusive/list_hook.h"
#include "oneflow/core/intrusive/struct_traits.h"

namespace oneflow {
namespace intrusive {

#define INTRUSIVE_FOR_EACH(elem, container) \
  _INTRUSIVE_FOR_EACH(std::remove_pointer<decltype(container)>::type, elem, container)

#define INTRUSIVE_FOR_EACH_PTR(elem, container) \
  _INTRUSIVE_FOR_EACH_PTR(std::remove_pointer<decltype(container)>::type, elem, container)

#define INTRUSIVE_UNSAFE_FOR_EACH_PTR(elem, container) \
  _INTRUSIVE_UNSAFE_FOR_EACH_PTR(std::remove_pointer<decltype(container)>::type, elem, container)

// details

#define _INTRUSIVE_FOR_EACH(container_type, elem, container)                     \
  for (intrusive::shared_ptr<typename container_type::value_type> elem,          \
       *end_if_not_null = nullptr;                                               \
       end_if_not_null == nullptr; end_if_not_null = nullptr, ++end_if_not_null) \
  LIST_HOOK_FOR_EACH_WITH_EXPR(                                                  \
      (intrusive::OffsetStructField<                                             \
          typename container_type, intrusive::ListHook,                          \
          container_type::IteratorHookOffset()>::FieldPtr4StructPtr(container)), \
      container_type::iterator_struct_field, elem_ptr, (elem.Reset(elem_ptr), true))

#define _INTRUSIVE_FOR_EACH_PTR(container_type, elem, container)                                \
  LIST_HOOK_FOR_EACH((intrusive::OffsetStructField<                                             \
                         typename container_type, intrusive::ListHook,                          \
                         container_type::IteratorHookOffset()>::FieldPtr4StructPtr(container)), \
                     container_type::iterator_struct_field, elem)

#define _INTRUSIVE_UNSAFE_FOR_EACH_PTR(container_type, elem, container)          \
  LIST_HOOK_UNSAFE_FOR_EACH(                                                     \
      (intrusive::OffsetStructField<                                             \
          typename container_type, intrusive::ListHook,                          \
          container_type::IteratorHookOffset()>::FieldPtr4StructPtr(container)), \
      container_type::iterator_struct_field, elem)

}  // namespace intrusive
}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_FOR_EACH_H_
