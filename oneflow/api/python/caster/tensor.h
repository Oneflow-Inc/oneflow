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
#ifndef ONEFLOW_API_PYTHON_CASTER_TENSOR_H_
#define ONEFLOW_API_PYTHON_CASTER_TENSOR_H_

#include <pybind11/pybind11.h>

#include "oneflow/api/python/caster/common.h"
#include "oneflow/api/python/framework/tensor.h"

namespace pybind11 {
namespace detail {

template<typename T>
struct tensor_type_caster {
 public:
  bool load(handle src, bool convert) {
    using namespace oneflow::one;
    value_ = nullptr;
    if (!src) { return false; }
    if (src.is_none()) { return true; }
    if (!PyTensor_Check(src.ptr())) { return false; }
    value_ = PyTensor_Unpack(src.ptr());
    return true;
  }

  template<typename U>
  static handle cast(U&& src, return_value_policy policy, handle parent) {
    using namespace oneflow::one;
    return reinterpret_steal<object>(PyTensor_New(std::const_pointer_cast<Tensor>(src))).release();
  }

  operator std::shared_ptr<T>*() { return &value_; }
  operator std::shared_ptr<T>&() { return value_; }
  operator std::shared_ptr<T>&&() && { return std::move(value_); }

  static constexpr auto name = _("tensor");
  template<typename U>
  using cast_op_type = pybind11::detail::cast_op_type<std::shared_ptr<T>>;

 protected:
  std::shared_ptr<T> value_;
};

template<typename T>
struct parameter_type_caster {
 public:
  bool load(handle src, bool convert) {
    using namespace oneflow::one;
    value_ = nullptr;
    if (!src) { return false; }
    if (src.is_none()) { return true; }
    if (!PyTensor_Check(src.ptr())) { return false; }
    value_ = PyTensor_Unpack(src.ptr());
    return true;
  }

  template<typename U>
  static handle cast(U&& src, return_value_policy policy, handle parent) {
    using namespace oneflow::one;
    return reinterpret_steal<object>(PyParameter_New(std::const_pointer_cast<Parameter>(src)))
        .release();
  }

  operator std::shared_ptr<T>*() { return &value_; }
  operator std::shared_ptr<T>&() { return value_; }
  operator std::shared_ptr<T>&&() && { return std::move(value_); }

  static constexpr auto name = _("parameter");
  template<typename U>
  using cast_op_type = pybind11::detail::cast_op_type<std::shared_ptr<T>>;

 protected:
  std::shared_ptr<T> value_;
};

template<>
struct type_caster<std::shared_ptr<oneflow::one::Tensor>>
    : public tensor_type_caster<oneflow::one::Tensor> {};
template<>
struct type_caster<std::shared_ptr<const oneflow::one::Tensor>>
    : public tensor_type_caster<const oneflow::one::Tensor> {};

template<>
struct type_caster<std::shared_ptr<oneflow::one::Parameter>>
    : public parameter_type_caster<oneflow::one::Parameter> {};
template<>
struct type_caster<std::shared_ptr<const oneflow::one::Parameter>>
    : public parameter_type_caster<const oneflow::one::Parameter> {};

}  // namespace detail
}  // namespace pybind11

#endif  // ONEFLOW_API_PYTHON_CASTER_TENSOR_H_
