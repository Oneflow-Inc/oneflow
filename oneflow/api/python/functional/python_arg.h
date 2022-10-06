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
#include <Python.h>

#include "oneflow/core/common/throw.h"
#include "oneflow/api/python/functional/value_types.h"
#include "oneflow/core/common/maybe.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

struct DefaultVal {
  virtual ValueType value_type() const = 0;
  virtual const void* Ptr() const = 0;
};

template<typename T>
struct TypedDefaultVal final : public DefaultVal {
  T content;
  explicit TypedDefaultVal(const T& v) : content(v) {}

  ValueType value_type() const override { return ValueTypeOf<T>(); }
  const void* Ptr() const override { return &content; }
};

template<typename T>
struct optional_traits {
  using type = void;
};

template<typename T>
struct optional_traits<Optional<T>> {
  using type =
      decltype(std::declval<Optional<T>>().Data_YouAreNotAllowedToCallThisFuncOutsideThisFile());
};

}  // namespace detail

class PythonArg {
 public:
  PythonArg() = default;

  PythonArg(PyObject* object, int size = 0)
      : object_(object), default_val_(), size_(size), tag_(HAS_OBJECT) {}

  PythonArg(const detail::DefaultVal* value, int size = 0)
      : object_(nullptr), default_val_(value), size_(size), tag_(HAS_DEFAULT) {}

  template<typename T, typename std::enable_if<!internal::IsOptional<T>::value, int>::type = 0>
  T As() const {
    if (tag_ == HAS_DEFAULT) {
      CHECK_EQ_OR_THROW(ValueTypeOf<T>(), default_val_->value_type())
          << "Could not convert default value from type " << default_val_->value_type()
          << " to type " << ValueTypeOf<T>();
      return *reinterpret_cast<const T*>(default_val_->Ptr());
    }
    CHECK_EQ_OR_THROW(tag_, HAS_OBJECT);
    return ObjectAs<oneflow::detail::remove_cvref_t<T>>();
  }

  template<typename T, typename std::enable_if<internal::IsOptional<T>::value, int>::type = 0>
  T As() const {
    if (tag_ == HAS_DEFAULT) {
      CHECK_EQ_OR_THROW(ValueTypeOf<T>(), default_val_->value_type())
          << "Could not convert default value from type " << default_val_->value_type()
          << " to type " << ValueTypeOf<T>();
      return *reinterpret_cast<const T*>(default_val_->Ptr());
    }
    CHECK_EQ_OR_THROW(tag_, HAS_OBJECT);
    if (object_ == Py_None) { return T(); }
    return ObjectAs<typename detail::optional_traits<T>::type>();
  }

  bool TypeCheck(ValueType type) const;

 private:
  template<typename T>
  T ObjectAs() const;

  PyObject* object_;
  const detail::DefaultVal* default_val_;
  size_t size_;
  enum { HAS_OBJECT, HAS_DEFAULT, HAS_NONE } tag_;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_
