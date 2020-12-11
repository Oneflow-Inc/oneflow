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
#ifndef ONEFLOW_CORE_COMMON_PTR_UTIL_H_
#define ONEFLOW_CORE_COMMON_PTR_UTIL_H_

#include <memory>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/type_traits.h"

namespace oneflow {

// shared_ptr or unique_ptr
template<typename T>
Maybe<scalar_or_const_ref_t<typename T::element_type>> PtrGet(const T& ptr) {
  CHECK_OR_RETURN(static_cast<bool>(ptr)) << "invalid nullptr";
  return *ptr;
}

template<typename T>
Maybe<scalar_or_const_ref_t<T>> PtrGet(T* ptr) {
  CHECK_N0TNULL_OR_RETURN(ptr);
  return *ptr;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_PTR_UTIL_H_
