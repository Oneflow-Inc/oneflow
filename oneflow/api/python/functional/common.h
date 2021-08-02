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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_

#include <string>
#include <vector>
#include <pybind11/pybind11.h>

#include "oneflow/api/python/framework/throw.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/functional/scalar.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

struct PyObjectPtrDeleter {
  inline void operator()(PyObject* obj) {
    if (obj) { Py_DECREF(obj); }
    obj = NULL;
  }
};

using PyObjectPtr = std::unique_ptr<PyObject, PyObjectPtrDeleter>;

#define ARITHMETIC_TYPE_SEQ      \
  OF_PP_MAKE_TUPLE_SEQ(int32_t)  \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t)  \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t) \
  OF_PP_MAKE_TUPLE_SEQ(float)    \
  OF_PP_MAKE_TUPLE_SEQ(double)   \
  OF_PP_MAKE_TUPLE_SEQ(bool)

#define MAKE_LIST_TUPLE_SEQ(T) (MAKE_LIST_TUPLE(T))
#define MAKE_LIST_TUPLE(T) (std::vector<T>)
#define ARITHMETIC_LIST_TYPE_SEQ OF_PP_FOR_EACH_TUPLE(MAKE_LIST_TUPLE_SEQ, ARITHMETIC_TYPE_SEQ)

template<typename T>
inline bool isinstance(py::handle obj) {
  return py::isinstance<T>(obj);
}

#define SPECIALIZE_IS_INSTANCE(T)                                                \
  template<>                                                                     \
  inline bool isinstance<T>(py::handle obj) {                                    \
    static py::object dummy = py::cast(T());                                     \
    CHECK_NOTNULL_OR_THROW(dummy.ptr()) << "Pybind has no internal type " << #T; \
    return py::isinstance(obj, py::type::of(dummy));                             \
  }

OF_PP_FOR_EACH_TUPLE(SPECIALIZE_IS_INSTANCE, ARITHMETIC_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(std::string));
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_IS_INSTANCE,
                     ARITHMETIC_LIST_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(std::vector<std::string>));

#undef SPECIALIZE_IS_INSTANCE

template<typename T>
struct type_caster {
  static Maybe<T> cast(py::handle src) { return py::cast<T>(src); }
};

template<typename T>
inline Maybe<T> cast(py::handle obj) {
  return type_caster<T>::cast(obj);
}

template<typename T>
struct type_caster<std::vector<T>> {
  static Maybe<std::vector<T>> cast(py::handle src);
};

#define SPECIALIZE_INTERNAL_IS_INSTANCE_ADN_CAST(T)                           \
  template<>                                                                  \
  inline bool isinstance<std::shared_ptr<T>>(py::handle obj) {                \
    return detail::isinstance<T>(obj);                                        \
  }                                                                           \
                                                                              \
  template<>                                                                  \
  struct type_caster<std::shared_ptr<T>> {                                    \
    static Maybe<std::shared_ptr<T>> cast(py::handle src) {                   \
      CHECK_OR_RETURN(detail::isinstance<T>(src))                             \
          << "Can not cast to " << #T << " from python object whose type is " \
          << *JUST(detail::cast<std::string>(py::str(py::type::of(src))));    \
      return py::cast<std::shared_ptr<T>>(src);                               \
    }                                                                         \
  };

SPECIALIZE_INTERNAL_IS_INSTANCE_ADN_CAST(one::Tensor);
SPECIALIZE_INTERNAL_IS_INSTANCE_ADN_CAST(one::TensorTuple);
SPECIALIZE_INTERNAL_IS_INSTANCE_ADN_CAST(MutableCfgAttrMap);

#undef SPECIALIZE_INTERNAL_IS_INSTANCE_ADN_CAST

template<typename T>
T&& dereference(T&& val) {
  return std::move(val);
}

template<typename T>
T&& dereference(std::shared_ptr<T>&& val) {
  return std::move(*val);
}

template<typename T>
/* static */ Maybe<std::vector<T>> type_caster<std::vector<T>>::cast(py::handle src) {
  PyObject* obj = src.ptr();
  bool is_tuple = PyTuple_Check(obj);
  CHECK_OR_RETURN(is_tuple || PyList_Check(obj))
      << "The python object is not list or tuple, but is "
      << *JUST(detail::cast<std::string>(py::str(py::type::of(src))));
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);

  auto values = std::make_shared<std::vector<T>>(size);
  for (int i = 0; i < size; ++i) {
    values->at(i) = detail::dereference<T>(JUST(detail::cast<T>(
        is_tuple ? py::handle(PyTuple_GET_ITEM(obj, i)) : py::handle(PyList_GET_ITEM(obj, i)))));
  }
  return values;
}

}  // namespace detail

bool PyTensorCheck(PyObject* object);
Maybe<Tensor> PyUnpackTensor(PyObject* object);

bool PyScalarCheck(PyObject* object);
Maybe<Scalar> PyUnpackScalar(PyObject* object);

const char* PyStringAsString(PyObject* object);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
