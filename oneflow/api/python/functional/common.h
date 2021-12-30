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

#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/tensor_index.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

struct PyObjectPtrDeleter {
  inline void operator()(PyObject* obj) {
    py::gil_scoped_acquire acquire;
    if (obj) { Py_DECREF(obj); }
    obj = NULL;
  }
};

using PyObjectPtr = std::unique_ptr<PyObject, PyObjectPtrDeleter>;

#define INTEGER_TYPE_SEQ         \
  OF_PP_MAKE_TUPLE_SEQ(int32_t)  \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t)  \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t) \
  OF_PP_MAKE_TUPLE_SEQ(bool)

#define FLOATING_TYPE_SEQ     \
  OF_PP_MAKE_TUPLE_SEQ(float) \
  OF_PP_MAKE_TUPLE_SEQ(double)

template<typename T>
T dereference(T&& val) {
  return std::forward<T>(val);
}

template<typename T>
T dereference(std::shared_ptr<T>&& val) {
  return *val;
}

bool PySequenceCheck(PyObject* obj);
bool PySequenceCheck(PyObject* obj, const std::function<bool(PyObject*)>& item_check);

template<typename T, typename UnpackItemFunc>
inline Maybe<std::vector<T>> PyUnpackSequence(PyObject* obj, UnpackItemFunc unpack_item) {
  bool is_tuple = PyTuple_Check(obj);
  CHECK_OR_RETURN(is_tuple || PyList_Check(obj))
      << "The object is not list or tuple, but is " << Py_TYPE(obj)->tp_name;
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  auto values = std::make_shared<std::vector<T>>(size);
  for (int i = 0; i < size; ++i) {
    PyObject* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    values->at(i) = dereference<T>(JUST(unpack_item(item)));
  }
  return values;
}

// Integer/Float list
bool PyLongSequenceCheck(PyObject* obj);
bool PyFloatSquenceCheck(PyObject* obj);

template<typename T>
inline Maybe<std::vector<T>> PyUnpackLongSequence(PyObject* obj) {
  return PyUnpackSequence<T>(
      obj, [](PyObject* item) -> Maybe<T> { return static_cast<T>(PyLong_AsLongLong(item)); });
}

template<typename T>
inline Maybe<std::vector<T>> PyUnpackFloatSequence(PyObject* obj) {
  return PyUnpackSequence<T>(
      obj, [](PyObject* item) -> Maybe<T> { return static_cast<T>(PyFloat_AsDouble(item)); });
}

// String
bool PyStringCheck(PyObject* obj);
bool PyStringSequenceCheck(PyObject* obj);

Maybe<const char*> PyStringAsString(PyObject* obj);

// Scalar
bool PyScalarCheck(PyObject* obj);
Maybe<Scalar> PyUnpackScalar(PyObject* obj);

// Tensor
bool PyTensorCheck(PyObject* obj);
Maybe<Tensor> PyUnpackTensor(PyObject* obj);

// Tensor list
bool PyTensorSequenceCheck(PyObject* obj);
Maybe<std::vector<std::shared_ptr<Tensor>>> PyUnpackTensorSequence(PyObject* obj);

// TensorTuple
bool PyTensorTupleCheck(PyObject* obj);
Maybe<TensorTuple> PyUnpackTensorTuple(PyObject* obj);

// DType
bool PyDTypeCheck(PyObject* obj);
Maybe<Symbol<DType>> PyUnpackDType(PyObject* obj);

// DType list
bool PyDTypeSequenceCheck(PyObject* obj);
Maybe<std::vector<Symbol<DType>>> PyUnpackDTypeSequence(PyObject* obj);

// Shape list
bool PyShapeSequenceCheck(PyObject* obj);
Maybe<std::vector<Shape>> PyUnpackShapeSequence(PyObject* obj);

// Generator
bool PyGeneratorCheck(PyObject* obj);
Maybe<Generator> PyUnpackGenerator(PyObject* obj);

// Device
bool PyDeviceCheck(PyObject* obj);
Maybe<Symbol<Device>> PyUnpackDevice(PyObject* obj);

// Placement
bool PyParallelDescCheck(PyObject* obj);
Maybe<Symbol<ParallelDesc>> PyUnpackParallelDesc(PyObject* obj);

// SBP
bool PySbpParallelCheck(PyObject* obj);
Maybe<Symbol<cfg::SbpParallel>> PyUnpackSbpParallel(PyObject* obj);

// SBP list
bool PySbpParallelSequenceCheck(PyObject* obj);
Maybe<std::vector<Symbol<cfg::SbpParallel>>> PyUnpackSbpParallelSequence(PyObject* obj);

// Tensor index
bool PyTensorIndexCheck(PyObject* obj);
Maybe<TensorIndex> PyUnpackTensorIndex(PyObject* obj);

// OpExpr
bool PyOpExprCheck(PyObject* obj);
Maybe<OpExpr> PyUnpackOpExpr(PyObject* obj);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
