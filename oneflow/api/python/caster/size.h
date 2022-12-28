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
#ifndef ONEFLOW_API_PYTHON_CASTER_SIZE_H_
#define ONEFLOW_API_PYTHON_CASTER_SIZE_H_
#include <type_traits>
#include <Python.h>
#include <pybind11/pybind11.h>

#include "oneflow/api/python/framework/size.h"
#include "oneflow/core/common/shape.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class shape : public object {
 public:
  PYBIND11_OBJECT_CVT(shape, object, oneflow::TensorSize_Check, raw_shape)
  explicit shape(size_t size = 0) : object(oneflow::TensorSize_New((ssize_t)size), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate tensor size object!");
  }
  size_t size() const { return (size_t)PyTuple_Size(m_ptr); }
  bool empty() const { return size() == 0; }
  detail::tuple_accessor operator[](size_t index) const { return {*this, index}; }
  detail::item_accessor operator[](handle h) const { return object::operator[](h); }
  detail::tuple_iterator begin() const { return {*this, 0}; }
  detail::tuple_iterator end() const { return {*this, PyTuple_GET_SIZE(m_ptr)}; }

 private:
  static PyObject* raw_shape(PyObject* op) {
    if (oneflow::TensorSize_Check(op)) return handle(op).inc_ref().ptr();
    return PyObject_CallFunctionObjArgs((PyObject*)&oneflow::TensorSize_Type, op, NULL);
  }
};

PYBIND11_NAMESPACE_BEGIN(detail)

template<typename T>
struct shape_type_caster {
 public:
  bool load(handle src, bool convert) {
    value_ = nullptr;
    if (src && src.is_none()) { return true; }
    if (!oneflow::TensorSize_Check(src.ptr())) { return false; }
    value_ = std::make_shared<T>(oneflow::TensorSize_AsShape(src.ptr()));
    return true;
  }

  template<typename U>
  static handle cast(U&& src, return_value_policy /*policy*/, handle /*parent*/) {
    return cast_impl(std::forward<U>(src));
  }

  template<typename U>
  static handle cast(U* src, return_value_policy policy, handle parent) {
    if (!src) { return none().release(); }
    return cast(*src, policy, parent);
  }

  operator T*() { return value_.get(); }
  operator T&() { return *value_; }
  operator T&&() && { return std::move(*value_); }

  operator std::shared_ptr<T>*() { return &value_; }
  operator std::shared_ptr<T>&() { return value_; }
  operator std::shared_ptr<T>&&() && { return std::move(value_); }

  static constexpr auto name = _("shape");
  template<typename U>
  using cast_op_type = pybind11::detail::cast_op_type<std::shared_ptr<T>>;

 private:
  static handle cast_impl(const oneflow::Shape& src) {
    return reinterpret_steal<shape>(oneflow::TensorSize_NewFromShape(src)).release();
  }
  static handle cast_impl(const std::shared_ptr<const oneflow::Shape>& src) {
    return reinterpret_steal<shape>(oneflow::TensorSize_NewFromShape(*src)).release();
  }

 protected:
  std::shared_ptr<T> value_;
};

template<>
struct type_caster<oneflow::Shape> : public shape_type_caster<oneflow::Shape> {};
template<>
struct type_caster<std::shared_ptr<oneflow::Shape>> : public shape_type_caster<oneflow::Shape> {};
template<>
struct type_caster<std::shared_ptr<const oneflow::Shape>>
    : public shape_type_caster<const oneflow::Shape> {};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#endif  // ONEFLOW_API_PYTHON_CASTER_SIZE_H_
