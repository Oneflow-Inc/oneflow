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
#include "oneflow/api/python/functional/value_types.h"
#include "oneflow/core/common/maybe.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

struct Immediate {
  virtual ValueType value_type() const = 0;
  virtual const void* Ptr() const = 0;
};

template<typename T>
struct TypedImmediate final : public Immediate {
  T content;
  explicit TypedImmediate(const T& v) : content(v) {}

  ValueType value_type() const override { return ValueTypeOf<T>(); }
  const void* Ptr() const override { return &content; }
};

}  // namespace detail

class PythonArg {
 public:
  PythonArg() = default;
  PythonArg(const py::object& object)
      : object_(object.ptr()), immediate_(), size_(0), active_tag_(HAS_OBJECT) {}

  PythonArg(const py::object& object, int size)
      : object_(object.ptr()), immediate_(), size_(size), active_tag_(HAS_OBJECT) {}

  PythonArg(const std::shared_ptr<const detail::Immediate>& value)
      : object_(nullptr), immediate_(value), size_(0), active_tag_(HAS_IMMEDIATE) {}

  template<typename T, typename std::enable_if<!py::detail::is_pyobject<T>::value, int>::type = 0>
  PythonArg(const T& value)
      : object_(nullptr),
        immediate_(std::make_shared<detail::TypedImmediate<T>>(value)),
        size_(0),
        active_tag_(HAS_IMMEDIATE) {}

  virtual ~PythonArg() = default;

  PythonArg(const PythonArg& other)
      : object_(other.object_),
        immediate_(other.immediate_),
        size_(other.size_),
        active_tag_(other.active_tag_) {}
  PythonArg(PythonArg&& other)
      : object_(other.object_),
        immediate_(std::move(other.immediate_)),
        size_(other.size_),
        active_tag_(other.active_tag_) {}

  PythonArg& operator=(const PythonArg& other) {
    object_ = other.object_;
    immediate_ = other.immediate_;
    size_ = other.size_;
    active_tag_ = other.active_tag_;
    return *this;
  }
  PythonArg& operator=(PythonArg&& other) {
    object_ = other.object_;
    immediate_ = std::move(other.immediate_);
    size_ = other.size_;
    active_tag_ = other.active_tag_;
    return *this;
  }

  template<typename T>
  struct ObjectAsHelper {
    Maybe<T> operator()(const PythonArg* self) { return self->ObjectAs<T>(); }
  };
  template<typename T>
  struct ObjectAsHelper<Optional<T>> {
    Maybe<Optional<T>> operator()(const PythonArg* self) {
      if (self->object_ == Py_None) { return std::make_shared<Optional<T>>(); }
      return std::make_shared<Optional<T>>(JUST(self->ObjectAs<T>()));
    }
  };

  template<typename T>
  T As() const {
    if (active_tag_ == HAS_IMMEDIATE) {
      CHECK_EQ_OR_THROW(ValueTypeOf<T>(), immediate_->value_type())
          << "Could not convert immediate value from type " << immediate_->value_type()
          << " to type " << ValueTypeOf<T>();
      return *reinterpret_cast<const T*>(immediate_->Ptr());
    }
    CHECK_EQ_OR_THROW(active_tag_, HAS_OBJECT);
    return ObjectAsHelper<oneflow::detail::remove_cvref_t<T>>()(this).GetOrThrow();
  }

  Maybe<bool> TypeCheck(ValueType type) const;

 private:
  template<typename T>
  Maybe<T> ObjectAs() const;

  PyObject* object_;
  std::shared_ptr<const detail::Immediate> immediate_;
  size_t size_;
  enum { HAS_OBJECT, HAS_IMMEDIATE, HAS_NONE } active_tag_;
};

bool PythonArgCheck(const PythonArg& arg, ValueType type);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_ARG_H_
