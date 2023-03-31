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
#include <complex>
#include <pybind11/pybind11.h>

#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/api/python/caster/maybe.h"
#include "oneflow/api/python/caster/optional.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/layout.h"
#include "oneflow/core/framework/memory_format.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/tensor_index.h"
#include "oneflow/core/common/foreign_lock_helper.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

struct PyObjectPtrDeleter {
  inline void operator()(PyObject* obj) {
    CHECK_JUST(Singleton<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
      if (obj) { Py_DECREF(obj); }
      obj = NULL;
      return Maybe<void>::Ok();
    }));
  }
};

using PyObjectPtr = std::unique_ptr<PyObject, PyObjectPtrDeleter>;

#define INTEGER_AND_BOOL_TYPE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(int32_t)   \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t)  \
  OF_PP_MAKE_TUPLE_SEQ(int64_t)   \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t)  \
  OF_PP_MAKE_TUPLE_SEQ(bool)

#define FLOATING_TYPE_SEQ     \
  OF_PP_MAKE_TUPLE_SEQ(float) \
  OF_PP_MAKE_TUPLE_SEQ(double)

bool PySequenceCheck(PyObject* obj);
bool PySequenceCheck(PyObject* obj, const std::function<bool(PyObject*)>& item_check);

template<typename T, typename UnpackItemFunc>
inline std::vector<T> PyUnpackSequence(PyObject* obj, UnpackItemFunc unpack_item) {
  bool is_tuple = PyTuple_Check(obj);
  CHECK_OR_THROW(is_tuple || PyList_Check(obj))
      << "The object is not list or tuple, but is " << Py_TYPE(obj)->tp_name;
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<T> values(size);
  for (int i = 0; i < size; ++i) {
    PyObject* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    values[i] = unpack_item(item);
  }
  return values;
}

// Scalar Tensor
bool PyScalarTensorCheck(PyObject* obj);
Scalar PyUnpackScalarTensor(PyObject* obj);

#define DefinePyTypeScalarTensorCheck(type, type_check_func)               \
  inline bool Py##type##ScalarTensorCheck(PyObject* obj) {                 \
    return PyScalarTensorCheck(obj)                                        \
           && type_check_func(PyTensor_Unpack(obj)->dtype()->data_type()); \
  }

DefinePyTypeScalarTensorCheck(Bool, IsBoolDataType);         // PyBoolScalarTensorCheck
DefinePyTypeScalarTensorCheck(Integer, IsIntegralDataType);  // PyIntegerScalarTensorCheck
DefinePyTypeScalarTensorCheck(Float, IsFloatingDataType);    // PyFloatScalarTensorCheck
DefinePyTypeScalarTensorCheck(Complex, IsComplexDataType);   // PyComplexScalarTensorCheck
#undef DefinePyTypeScalarTensorCheck

bool PyUnpackBoolScalarTensor(PyObject* obj);
long long PyUnpackIntegerScalarTensor_AsLongLong(PyObject* obj);
double PyUnpackFloatScalarTensor_AsDouble(PyObject* obj);
std::complex<double> PyUnpackComplexScalarTensor_AsCComplex(PyObject* obj);

// Integer/Float list
bool PyLongSequenceCheck(PyObject* obj);
bool PyFloatSequenceCheck(PyObject* obj);

template<typename T>
inline std::vector<T> PyUnpackLongSequence(PyObject* obj) {
  return PyUnpackSequence<T>(obj, [](PyObject* item) -> T {
    if (PyIntegerScalarTensorCheck(item)) {
      return static_cast<T>(PyUnpackIntegerScalarTensor_AsLongLong(item));
    }
    return static_cast<T>(PyLong_AsLongLong(item));
  });
}

template<typename T>
inline std::vector<T> PyUnpackFloatSequence(PyObject* obj) {
  return PyUnpackSequence<T>(obj, [](PyObject* item) -> T {
    if (PyFloatScalarTensorCheck(item)) {
      return static_cast<T>(PyUnpackFloatScalarTensor_AsDouble(item));
    }
    return static_cast<T>(PyFloat_AsDouble(item));
  });
}

// String
bool PyStringCheck(PyObject* obj);
bool PyStringSequenceCheck(PyObject* obj);

std::string PyStringAsString(PyObject* obj);

std::string PyObjectToReprStr(PyObject* obj);

// Scalar
bool PyScalarCheck(PyObject* obj);
Scalar PyUnpackScalar(PyObject* obj);

// Tensor list
bool PyTensorSequenceCheck(PyObject* obj);
std::vector<std::shared_ptr<Tensor>> PyUnpackTensorSequence(PyObject* obj);

// TensorTuple
bool PyTensorTupleCheck(PyObject* obj);
std::shared_ptr<TensorTuple> PyUnpackTensorTuple(PyObject* obj);

// DType
bool PyDTypeCheck(PyObject* obj);
Symbol<DType> PyUnpackDType(PyObject* obj);

// Layout
bool PyLayoutCheck(PyObject* obj);
Symbol<Layout> PyUnpackLayout(PyObject* obj);

// Memory Format
bool PyMemoryFormatCheck(PyObject* obj);
Symbol<MemoryFormat> PyUnpackMemoryFormat(PyObject* obj);

// DType list
bool PyDTypeSequenceCheck(PyObject* obj);
std::vector<Symbol<DType>> PyUnpackDTypeSequence(PyObject* obj);

// Shape
bool PyShapeCheck(PyObject* obj);
Shape PyUnpackShape(PyObject* obj);

// Shape list
bool PyShapeSequenceCheck(PyObject* obj);
std::vector<Shape> PyUnpackShapeSequence(PyObject* obj);

// Generator
bool PyGeneratorCheck(PyObject* obj);
std::shared_ptr<Generator> PyUnpackGenerator(PyObject* obj);

// Device
bool PyDeviceCheck(PyObject* obj);
Symbol<Device> PyUnpackDevice(PyObject* obj);

// Placement
bool PyParallelDescCheck(PyObject* obj);
Symbol<ParallelDesc> PyUnpackParallelDesc(PyObject* obj);

// SBP
bool PySbpParallelCheck(PyObject* obj);
Symbol<SbpParallel> PyUnpackSbpParallel(PyObject* obj);

// SBP list
bool PySbpParallelSequenceCheck(PyObject* obj);
std::vector<Symbol<SbpParallel>> PyUnpackSbpParallelSequence(PyObject* obj);

// Tensor index
bool PyTensorIndexCheck(PyObject* obj);
TensorIndex PyUnpackTensorIndex(PyObject* obj);

// OpExpr
bool PyOpExprCheck(PyObject* obj);
std::shared_ptr<OpExpr> PyUnpackOpExpr(PyObject* obj);

template<typename T>
inline PyObject* CastToPyObject(T&& t) {
  return py::cast(t).inc_ref().ptr();
}

template<>
inline PyObject* CastToPyObject<Maybe<Tensor>>(Maybe<Tensor>&& t) {
  return PyTensor_New(t.GetPtrOrThrow());
}

template<>
inline PyObject* CastToPyObject<Maybe<TensorTuple>>(Maybe<TensorTuple>&& t) {
  const auto& tensor_tuple = t.GetPtrOrThrow();
  py::tuple tup(tensor_tuple->size());
  for (int i = 0; i < tensor_tuple->size(); ++i) { tup[i] = py::cast(tensor_tuple->at(i)); }
  return py::cast<py::object>(tup).inc_ref().ptr();
}

template<>
inline PyObject* CastToPyObject<Maybe<void>>(Maybe<void>&& t) {
  t.GetOrThrow();
  Py_RETURN_NONE;
}

// int64_t
Maybe<int64_t> PyUnpackLong(PyObject* py_obj);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
