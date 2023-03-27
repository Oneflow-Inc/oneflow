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
#include "oneflow/api/python/functional/common.h"
#include <object.h>
#include <string>
#include <complex>

#include "oneflow/api/python/functional/indexing.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/functional/tensor_index.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/framework/tensor_util.h"
namespace oneflow {
namespace one {
namespace functional {

namespace detail {

namespace {

template<typename T>
Maybe<T> GetItemInPyScalarTensor(PyObject* obj) {
  return GetItemInScalarTensor<T>(PyTensor_Unpack(obj));
}

}  // namespace

template<typename T, typename std::enable_if<!std::is_base_of<py::object, T>::value, int>::type = 0>
bool isinstance_fast(PyObject* obj) {
  static auto type = py::detail::get_type_handle(typeid(T), false);
  if (!type) { return false; }
  const auto result = PyObject_IsInstance(obj, type.ptr());
  if (result == -1) { throw py::error_already_set(); }
  return result != 0;
}

template<typename T, typename std::enable_if<!std::is_base_of<py::object, T>::value
                                                 && !py::detail::is_shared_ptr<T>::value,
                                             int>::type = 0>
const T& cast_fast(PyObject* obj) {
  auto vh = reinterpret_cast<py::detail::instance*>(obj)->get_value_and_holder();
  auto*& vptr = vh.value_ptr();
  if (!vptr) {
    throw py::cast_error("Unable to cast from object to T& since lazy allocation is not allowed "
                         "for fast cast, please use pybind11::cast instead");
  }
  return *reinterpret_cast<T*>(&vptr);
}

template<typename T, typename std::enable_if<!std::is_base_of<py::object, T>::value
                                                 && py::detail::is_shared_ptr<T>::value,
                                             int>::type = 0>
const T& cast_fast(PyObject* obj) {
  auto vh = reinterpret_cast<py::detail::instance*>(obj)->get_value_and_holder();
  if (!vh.holder_constructed()) {
    throw py::cast_error("Unable to cast from non-held to held instance (T& to Holder<T>)");
  }
  return vh.template holder<T>();
}

}  // namespace detail

bool PySequenceCheck(PyObject* obj, const std::function<bool(PyObject*)>& item_check) {
  bool is_tuple = PyTuple_Check(obj);
  if (!is_tuple && !PyList_Check(obj)) { return false; }
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  if (size == 0) { return true; }
  PyObject* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) : PyList_GET_ITEM(obj, 0);
  return item_check(item);
}

bool PyLongSequenceCheck(PyObject* obj) {
  return PySequenceCheck(
      obj, [](PyObject* item) { return PyLong_Check(item) || PyIntegerScalarTensorCheck(item); });
}

bool PyFloatSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) {
    return PyFloat_Check(item) || PyLong_Check(item) || PyFloatScalarTensorCheck(item)
           || PyIntegerScalarTensorCheck(item);
  });
}

bool PyStringCheck(PyObject* obj) { return PyBytes_Check(obj) || PyUnicode_Check(obj); }

bool PyStringSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyStringCheck(item); });
}

std::string PyStringAsString(PyObject* obj) {
  PyObject* bytes = PyUnicode_AsEncodedString(obj, "utf-8", "~E~");
  std::string str = PyBytes_AS_STRING(bytes);
  Py_XDECREF(bytes);
  return str;
}

std::string PyObjectToReprStr(PyObject* obj) {
  PyObject* repr_obj = PyObject_Repr(obj);
  std::string str = PyStringAsString(repr_obj);
  Py_XDECREF(repr_obj);
  return str;
}

// Tensor list
bool PyTensorSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyTensor_Check(item); });
}
std::vector<std::shared_ptr<Tensor>> PyUnpackTensorSequence(PyObject* obj) {
  return PyUnpackSequence<std::shared_ptr<Tensor>>(
      obj, [](PyObject* item) { return PyTensor_Unpack(item); });
}

// TensorTuple
bool PyTensorTupleCheck(PyObject* obj) { return detail::isinstance_fast<TensorTuple>(obj); }

std::shared_ptr<TensorTuple> PyUnpackTensorTuple(PyObject* obj) {
  return detail::cast_fast<std::shared_ptr<TensorTuple>>(obj);
}

// Scalar
bool PyScalarCheck(PyObject* obj) {
  return PyLong_Check(obj) || PyFloat_Check(obj) || PyComplex_Check(obj);
}

Scalar PyUnpackScalar(PyObject* obj) {
  if (PyBool_Check(obj)) {
    return obj == Py_True;
  } else if (PyLong_Check(obj)) {
    return static_cast<int64_t>(PyLong_AsLongLong(obj));
  } else if (PyFloat_Check(obj)) {
    return PyFloat_AsDouble(obj);
  } else if (PyComplex_Check(obj)) {
    Py_complex value = PyComplex_AsCComplex(obj);
    return std::complex<double>{value.real, value.imag};
  } else if (PyArray_IsScalar(obj, Bool)) {
    return obj == Py_True;
  } else if (PyArray_IsScalar(obj, Floating)) {
    return PyFloat_AsDouble(obj);
  } else if (PyArray_IsScalar(obj, Complex64) || PyArray_IsScalar(obj, Complex128)) {
    Py_complex value = PyComplex_AsCComplex(obj);
    return std::complex<double>{value.real, value.imag};
  }
  THROW(RuntimeError) << "The object is not scalar, but is " << Py_TYPE(obj)->tp_name;
  return 0;
}

// Scalar Tensor
bool PyScalarTensorCheck(PyObject* obj) {
  if (!LazyMode::is_enabled() && PyTensor_Check(obj)) {
    const auto& tensor = PyTensor_Unpack(obj);
    return tensor->shape()->size() == 0
           && IsTriviallyCopyableDataType(tensor->dtype()->data_type());
  }
  return false;
}

Scalar PyUnpackScalarTensor(PyObject* obj) {
  if (PyBoolScalarTensorCheck(obj)) {
    return PyUnpackBoolScalarTensor(obj);
  } else if (PyIntegerScalarTensorCheck(obj)) {
    return PyUnpackIntegerScalarTensor_AsLongLong(obj);
  } else if (PyFloatScalarTensorCheck(obj)) {
    return PyUnpackFloatScalarTensor_AsDouble(obj);
  } else if (PyComplexScalarTensorCheck(obj)) {
    return PyUnpackComplexScalarTensor_AsCComplex(obj);
  }
  THROW(RuntimeError) << "The object is not scalar tensor, but is " << Py_TYPE(obj)->tp_name
                      << "with data type: "
                      << DataType_Name(PyTensor_Unpack(obj)->dtype()->data_type());
  return 0;
}

#define SWITCH_SCALAR_TENSOR_TO_SCALAR(cpp_type, of_type) \
  case of_type:                                           \
    return detail::GetItemInPyScalarTensor<cpp_type>(obj).GetOrThrow();

#define SCALAR_TENSOR_UNPACK_FUNC_IMPL(func_name, return_type, type_seq)                  \
  return_type func_name(PyObject* obj) {                                                  \
    const auto& tensor = PyTensor_Unpack(obj);                                            \
    DataType data_type = tensor->dtype()->data_type();                                    \
    switch (data_type) {                                                                  \
      OF_PP_FOR_EACH_TUPLE(SWITCH_SCALAR_TENSOR_TO_SCALAR, type_seq)                      \
      default: {                                                                          \
        throw py::cast_error("Cannot get ##cpp##type from scalar tensor with data type: " \
                             + DataType_Name(data_type));                                 \
      }                                                                                   \
    }                                                                                     \
  }

SCALAR_TENSOR_UNPACK_FUNC_IMPL(PyUnpackBoolScalarTensor, bool,
                               BOOL_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ);
SCALAR_TENSOR_UNPACK_FUNC_IMPL(PyUnpackIntegerScalarTensor_AsLongLong, long long,
                               INT_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ
                                   CHAR_DATA_TYPE_SEQ);
SCALAR_TENSOR_UNPACK_FUNC_IMPL(PyUnpackFloatScalarTensor_AsDouble, double,
                               FLOATING_DATA_TYPE_SEQ INT_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ);
SCALAR_TENSOR_UNPACK_FUNC_IMPL(PyUnpackComplexScalarTensor_AsCComplex, std::complex<double>,
                               COMPLEX_DATA_TYPE_SEQ FLOATING_DATA_TYPE_SEQ INT_DATA_TYPE_SEQ
                                   UNSIGNED_INT_DATA_TYPE_SEQ);
#undef SWITCH_SCALAR_TENSOR_TO_SCALAR
#undef SCALAR_TENSOR_UNPACK_FUNC_IMPL

// DType
bool PyDTypeCheck(PyObject* obj) { return detail::isinstance_fast<Symbol<DType>>(obj); }
Symbol<DType> PyUnpackDType(PyObject* obj) { return *detail::cast_fast<Symbol<DType>*>(obj); }

// Layout
bool PyLayoutCheck(PyObject* obj) { return detail::isinstance_fast<Symbol<Layout>>(obj); }
Symbol<Layout> PyUnpackLayout(PyObject* obj) { return *detail::cast_fast<Symbol<Layout>*>(obj); }

// Memory Format
bool PyMemoryFormatCheck(PyObject* obj) {
  return detail::isinstance_fast<Symbol<MemoryFormat>>(obj);
}
Symbol<MemoryFormat> PyUnpackMemoryFormat(PyObject* obj) {
  return *detail::cast_fast<Symbol<MemoryFormat>*>(obj);
}

// DType list
bool PyDTypeSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyDTypeCheck(item); });
}
std::vector<Symbol<DType>> PyUnpackDTypeSequence(PyObject* obj) {
  return PyUnpackSequence<Symbol<DType>>(obj, [](PyObject* item) { return PyUnpackDType(item); });
}

// Shape
bool PyShapeCheck(PyObject* obj) { return PyLongSequenceCheck(obj); }

Shape PyUnpackShape(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  CHECK_OR_THROW(is_tuple || PyList_Check(obj))
      << "The object is not list or tuple, but is " << Py_TYPE(obj)->tp_name;
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  DimVector values(size);
  for (int i = 0; i < size; ++i) {
    PyObject* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    values[i] = PyLong_AsLongLong(item);
  }
  return Shape(values);
}

// Shape list
bool PyShapeSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyLongSequenceCheck(item); });
}
std::vector<Shape> PyUnpackShapeSequence(PyObject* obj) {
  return PyUnpackSequence<Shape>(obj, [](PyObject* item) -> Shape { return PyUnpackShape(item); });
}

// Generator
bool PyGeneratorCheck(PyObject* obj) { return detail::isinstance_fast<Generator>(obj); }
std::shared_ptr<Generator> PyUnpackGenerator(PyObject* obj) {
  return detail::cast_fast<std::shared_ptr<one::Generator>>(obj);
}

// Device
bool PyDeviceCheck(PyObject* obj) { return detail::isinstance_fast<Symbol<Device>>(obj); }
Symbol<Device> PyUnpackDevice(PyObject* obj) {
  return *detail::cast_fast<std::shared_ptr<Symbol<Device>>>(obj);
}

// Placement
bool PyParallelDescCheck(PyObject* obj) {
  return detail::isinstance_fast<Symbol<ParallelDesc>>(obj);
}
Symbol<ParallelDesc> PyUnpackParallelDesc(PyObject* obj) {
  return *detail::cast_fast<std::shared_ptr<Symbol<ParallelDesc>>>(obj);
}

// SBP
bool PySbpParallelCheck(PyObject* obj) { return detail::isinstance_fast<Symbol<SbpParallel>>(obj); }
Symbol<SbpParallel> PyUnpackSbpParallel(PyObject* obj) {
  return *detail::cast_fast<std::shared_ptr<Symbol<SbpParallel>>>(obj);
}

// SBP list
bool PySbpParallelSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PySbpParallelCheck(item); });
}
std::vector<Symbol<SbpParallel>> PyUnpackSbpParallelSequence(PyObject* obj) {
  return PyUnpackSequence<Symbol<SbpParallel>>(
      obj, [](PyObject* item) { return PyUnpackSbpParallel(item); });
}

// Tensor index
bool PyTensorIndexCheck(PyObject* obj) {
  return PySlice_Check(obj) || PyLong_Check(obj) || obj == Py_Ellipsis || obj == Py_None
         || PyTensor_Check(obj) || PySequence_Check(obj) || PyUnicode_Check(obj)
         || numpy::PyArrayCheckLongScalar(obj);
}
TensorIndex PyUnpackTensorIndex(PyObject* obj) {
  TensorIndex tensor_index;
  // Obvious single-entry cases.
  if (PySlice_Check(obj)                     // NOLINT
      || PyLong_Check(obj)                   // NOLINT
      || obj == Py_Ellipsis                  // NOLINT
      || obj == Py_None                      // NOLINT
      || PyTensor_Check(obj)                 // NOLINT
      || !PySequence_Check(obj)              // NOLINT
      || numpy::PyArrayCheckLongScalar(obj)  // NOLINT
      || PyUnicode_Check(obj)) {
    tensor_index.emplace_back(detail::UnpackIndexItem(obj));
    return tensor_index;
  }
  PyObject* tup = NULL;
  Py_ssize_t n = 0;
  if (PyTuple_Check(obj)) {
    tup = PySequence_Tuple(obj);
    n = PySequence_Size(tup);
  } else {
    // The follow comments are from numpy:
    // https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/mapping.c#L266
    /*
     * At this point, we're left with a non-tuple, non-array, sequence:
     * typically, a list. We use some somewhat-arbitrary heuristics from here
     * onwards to decided whether to treat that list as a single index, or a
     * list of indices.
     */
    n = PySequence_Size(obj);
    // Negative size indicates a Python error in the PySequence_Size call.
    if (n < 0) {
      PyErr_Clear();
      tensor_index.emplace_back(detail::UnpackIndexItem(obj));
      return tensor_index;
    }
    // The follow comments are from numpy:
    // https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/mapping.c#L280
    /*
     * Backwards compatibility only takes effect for short sequences - otherwise
     * we treat it like any other scalar.
     *
     * Sequences < NPY_MAXDIMS with any slice objects
     * or newaxis, Ellipsis or other arrays or sequences
     * embedded, are considered equivalent to an indexing
     * tuple. (`a[[[1,2], [3,4]]] == a[[1,2], [3,4]]`)
     */
    if (n >= /*NPY_MAXDIMS=*/32) {
      tensor_index.emplace_back(detail::UnpackIndexItem(obj));
      return tensor_index;
    }
    // Check whether we should unpack the index like a tuple.
    bool commit_to_unpack = false;
    for (Py_ssize_t i = 0; i < n; ++i) {
      PyObject* item = PySequence_GetItem(obj, i);
      if (commit_to_unpack) {
        CHECK_OR_THROW(item) << "Sequence index is required.";
      } else {
        if (!item) {
          PyErr_Clear();
          break;
        }
        if (PySequence_Check(item)   // NOLINT
            || PySlice_Check(item)   // NOLINT
            || PyTensor_Check(item)  // NOLINT
            || item == Py_Ellipsis || item == Py_None) {
          commit_to_unpack = true;
        }
      }
      Py_DECREF(item);
    }
    if (commit_to_unpack) {
      tup = PySequence_Tuple(obj);
    } else {
      tensor_index.emplace_back(detail::UnpackIndexItem(obj));
      return tensor_index;
    }
  }

  tensor_index.resize(n);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_GetItem(tup, i);
    tensor_index[i] = detail::UnpackIndexItem(item);
    Py_DECREF(item);
  }
  Py_DECREF(tup);
  return tensor_index;
}

// OpExpr
bool PyOpExprCheck(PyObject* obj) { return detail::isinstance_fast<OpExpr>(obj); }

std::shared_ptr<OpExpr> PyUnpackOpExpr(PyObject* obj) {
  return detail::cast_fast<std::shared_ptr<OpExpr>>(obj);
}

// int64_t
Maybe<int64_t> PyUnpackLong(PyObject* py_obj) {
  int overflow = -1;
  long long val = PyLong_AsLongLongAndOverflow(py_obj, &overflow);
  if (val == -1 && PyErr_Occurred()) { return Error::RuntimeError() << "Python exception occurs"; }
  if (overflow != 0) { return Error::RuntimeError() << "Overflow when unpacking long"; }
  return (int64_t)val;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
