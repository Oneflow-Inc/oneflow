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

#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_

#include <pybind11/pybind11.h>

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/api/python/functional/common.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

struct Value {
  virtual ValueType type() const = 0;
  virtual const void* Ptr() const = 0;
};

template<typename T>
struct ValueImpl : public Value {
  T content;
  explicit ValueImpl(const T& v) : content(v) {}

  ValueType type() const override { return ValueTypeOf<T>(); }
  const void* Ptr() const override { return &content; }
};

}  // namespace detail

class PythonArg {
 public:
  PythonArg() = default;
  PythonArg(py::object object) : object_(object.ptr()), active_tag_(HAS_OBJECT) {}

  PythonArg(const std::shared_ptr<const detail::Value>& value)
      : value_(value), active_tag_(HAS_VALUE) {}

  template<typename T, typename std::enable_if<!py::detail::is_pyobject<T>::value, int>::type = 0>
  PythonArg(const T& value)
      : value_(std::make_shared<detail::ValueImpl<T>>(value)), active_tag_(HAS_VALUE) {}

  virtual ~PythonArg() = default;

  template<typename T>
  operator T() const {
    if (active_tag_ == HAS_VALUE) {
      CHECK_EQ(ValueTypeOf<T>(), value_->type())
          << "Could not convert value from type " << value_->type() << " to type "
          << ValueTypeOf<T>();
      return *reinterpret_cast<const T*>(value_->Ptr());
    }
    CHECK_EQ(active_tag_, HAS_OBJECT);
    return this->ObjectAs<  // NOLINT
        typename std::remove_cv<typename std::remove_reference<T>::type>::type>();
  }

 private:
  template<typename T>
  T ObjectAs() const;
  py::object Borrow() const { return py::reinterpret_borrow<py::object>(object_); }

  PyObject* object_;

  std::shared_ptr<const detail::Value> value_;
  enum { HAS_OBJECT, HAS_VALUE, HAS_NONE } active_tag_;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_
