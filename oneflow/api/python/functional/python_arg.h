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

#include "oneflow/api/python/framework/throw.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/functional/value_types.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

struct AnyDataBase {
  virtual ValueType value_type() const = 0;
  virtual const void* Ptr() const = 0;
};

template<typename T>
struct AnyData : public AnyDataBase {
  T content;
  explicit AnyData(const T& v) : content(v) {}

  ValueType value_type() const override { return ValueTypeOf<T>(); }
  const void* Ptr() const override { return &content; }
};

}  // namespace detail

class PythonArg {
 public:
  PythonArg() = default;
  PythonArg(py::object object) : object_(object.ptr()), active_tag_(HAS_OBJECT) {}

  PythonArg(const std::shared_ptr<const detail::AnyDataBase>& value)
      : immediate_(value), active_tag_(HAS_IMMEDIATE) {}

  template<typename T, typename std::enable_if<!py::detail::is_pyobject<T>::value, int>::type = 0>
  PythonArg(const T& value)
      : immediate_(std::make_shared<detail::AnyData<T>>(value)), active_tag_(HAS_IMMEDIATE) {}

  virtual ~PythonArg() = default;

  template<typename T>
  operator T() const {
    if (active_tag_ == HAS_IMMEDIATE) {
      CHECK_EQ_OR_THROW(ValueTypeOf<T>(), immediate_->value_type())
          << "Could not convert immediate value from type " << immediate_->value_type()
          << " to type " << ValueTypeOf<T>();
      return *reinterpret_cast<const T*>(immediate_->Ptr());
    }
    CHECK_EQ_OR_THROW(active_tag_, HAS_OBJECT);
    return this->ObjectAs<oneflow::detail::remove_cvref_t<T>>().GetOrThrow();
  }

 private:
  template<typename T>
  Maybe<T> ObjectAs() const;
  py::object Borrow() const { return py::reinterpret_borrow<py::object>(object_); }

  PyObject* object_;
  std::shared_ptr<const detail::AnyDataBase> immediate_;

  enum { HAS_OBJECT, HAS_IMMEDIATE, HAS_NONE } active_tag_;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_
