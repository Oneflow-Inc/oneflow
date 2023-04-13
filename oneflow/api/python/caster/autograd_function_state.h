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
#ifndef ONEFLOW_API_PYTHON_CASTER_AUTOGRAD_FUNCTION_STATE_H_
#define ONEFLOW_API_PYTHON_CASTER_AUTOGRAD_FUNCTION_STATE_H_

#include <pybind11/pybind11.h>

#include "oneflow/api/python/caster/common.h"
#include "oneflow/api/python/autograd/autograd_function_state.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template<typename T>
struct autograd_function_state_type_caster {
 public:
  bool load(handle src, bool convert) {
    using namespace oneflow::one;
    value_ = nullptr;
    if (!src) { return false; }
    if (src.is_none()) { return true; }
    if (!PyAutogradFunctionState_Check(src.ptr())) { return false; }
    value_ = ((PyAutogradFunctionState*)src.ptr())->data;
    return true;
  }

  template<typename U>
  static handle cast(U&& src, return_value_policy policy, handle parent) {
    using namespace oneflow::one;
    return reinterpret_steal<object>(
               PyAutogradFunctionState_NewFromPtr(
                   std::const_pointer_cast<FunctionAutoGradCaptureState>(src)))
        .release();
  }

  operator std::shared_ptr<T>*() { return &value_; }
  operator std::shared_ptr<T>&() { return value_; }
  operator std::shared_ptr<T>&&() && { return std::move(value_); }

  static constexpr auto name = _("autograd_function_state");

 protected:
  std::shared_ptr<T> value_;
};

template<>
struct type_caster<std::shared_ptr<oneflow::one::FunctionAutoGradCaptureState>>
    : public autograd_function_state_type_caster<oneflow::one::FunctionAutoGradCaptureState> {};
template<>
struct type_caster<std::shared_ptr<const oneflow::one::FunctionAutoGradCaptureState>>
    : public autograd_function_state_type_caster<const oneflow::one::FunctionAutoGradCaptureState> {
};

}  // namespace detail
}  // namespace pybind11

#endif  // ONEFLOW_API_PYTHON_CASTER_AUTOGRAD_FUNCTION_STATE_H_
