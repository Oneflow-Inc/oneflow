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

#include "oneflow/api/python/functional/indexing.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/tensor_index.h"

namespace oneflow {
namespace one {
namespace functional {

bool PySequenceCheck(PyObject* obj, const std::function<bool(PyObject*)>& item_check) {
  bool is_tuple = PyTuple_Check(obj);
  if (!is_tuple && !PyList_Check(obj)) { return false; }
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  if (size == 0) { return true; }
  PyObject* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) : PyList_GET_ITEM(obj, 0);
  return item_check(item);
}

bool PyLongSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyLong_Check(item); });
}

bool PyFloatSquenceCheck(PyObject* obj) {
  return PySequenceCheck(obj,
                         [](PyObject* item) { return PyFloat_Check(item) || PyLong_Check(item); });
}

bool PyStringCheck(PyObject* obj) { return PyBytes_Check(obj) || PyUnicode_Check(obj); }

bool PyStringSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyStringCheck(item); });
}

Maybe<std::string> PyStringAsString(PyObject* str_obj) {
  PyObject* bytes = PyUnicode_AsEncodedString(str_obj, "utf-8", "~E~");
  std::string str = PyBytes_AS_STRING(bytes);
  Py_XDECREF(bytes);
  return str;
}

Maybe<std::string> PyObjectToReprStr(PyObject* obj) {
  PyObject* repr_obj = PyObject_Repr(obj);
  std::string str = *JUST(PyStringAsString(repr_obj));
  Py_XDECREF(repr_obj);
  return str;
}

bool PyTensorCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<Tensor>(handle);
}

Maybe<Tensor> PyUnpackTensor(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::cast<std::shared_ptr<Tensor>>(handle);
}

// Tensor list
bool PyTensorSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyTensorCheck(item); });
}
Maybe<std::vector<std::shared_ptr<Tensor>>> PyUnpackTensorSequence(PyObject* obj) {
  return PyUnpackSequence<std::shared_ptr<Tensor>>(
      obj, [](PyObject* item) { return PyUnpackTensor(item); });
}

// TensorTuple
bool PyTensorTupleCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<TensorTuple>(handle);
}

Maybe<TensorTuple> PyUnpackTensorTuple(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::cast<std::shared_ptr<TensorTuple>>(handle);
}

// Scalar
bool PyScalarCheck(PyObject* obj) { return PyLong_Check(obj) || PyFloat_Check(obj); }

Maybe<Scalar> PyUnpackScalar(PyObject* obj) {
  if (PyBool_Check(obj)) {
    return std::make_shared<Scalar>(obj == Py_True);
  } else if (PyLong_Check(obj)) {
    return std::make_shared<Scalar>(static_cast<int64_t>(PyLong_AsLongLong(obj)));
  } else if (PyFloat_Check(obj)) {
    return std::make_shared<Scalar>(PyFloat_AsDouble(obj));
  }
  UNIMPLEMENTED_THEN_RETURN() << "The object is not scalar, but is " << Py_TYPE(obj)->tp_name;
}

// DType
bool PyDTypeCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<Symbol<DType>>(handle);
}
Maybe<Symbol<DType>> PyUnpackDType(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return *py::cast<Symbol<DType>*>(handle);
}

// DType list
bool PyDTypeSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyDTypeCheck(item); });
}
Maybe<std::vector<Symbol<DType>>> PyUnpackDTypeSequence(PyObject* obj) {
  return PyUnpackSequence<Symbol<DType>>(obj, [](PyObject* item) { return PyUnpackDType(item); });
}

// Shape list
bool PyShapeSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PyLongSequenceCheck(item); });
}
Maybe<std::vector<Shape>> PyUnpackShapeSequence(PyObject* obj) {
  return PyUnpackSequence<Shape>(obj, [](PyObject* item) -> Maybe<Shape> {
    const auto& shape = JUST(PyUnpackLongSequence<int64_t>(item));
    return std::make_shared<Shape>(DimVector(shape->begin(), shape->end()));
  });
}

// Generator
bool PyGeneratorCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<Generator>(handle);
}
Maybe<Generator> PyUnpackGenerator(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::cast<std::shared_ptr<one::Generator>>(handle);
}

// Device
bool PyDeviceCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<Symbol<Device>>(handle);
}
Maybe<Symbol<Device>> PyUnpackDevice(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return *py::cast<std::shared_ptr<Symbol<Device>>>(handle);
}

// Placement
bool PyParallelDescCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<Symbol<ParallelDesc>>(handle);
}
Maybe<Symbol<ParallelDesc>> PyUnpackParallelDesc(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return *py::cast<std::shared_ptr<Symbol<ParallelDesc>>>(handle);
}

// SBP
bool PySbpParallelCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<Symbol<SbpParallel>>(handle);
}
Maybe<Symbol<SbpParallel>> PyUnpackSbpParallel(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return *py::cast<std::shared_ptr<Symbol<SbpParallel>>>(handle);
}

// SBP list
bool PySbpParallelSequenceCheck(PyObject* obj) {
  return PySequenceCheck(obj, [](PyObject* item) { return PySbpParallelCheck(item); });
}
Maybe<std::vector<Symbol<SbpParallel>>> PyUnpackSbpParallelSequence(PyObject* obj) {
  return PyUnpackSequence<Symbol<SbpParallel>>(
      obj, [](PyObject* item) { return PyUnpackSbpParallel(item); });
}

// Tensor index
bool PyTensorIndexCheck(PyObject* obj) {
  return PySlice_Check(obj) || PyLong_Check(obj) || obj == Py_Ellipsis || obj == Py_None
         || PyTensorCheck(obj) || PySequence_Check(obj) || PyUnicode_Check(obj);
}
Maybe<TensorIndex> PyUnpackTensorIndex(PyObject* obj) {
  auto tensor_index = std::make_shared<TensorIndex>();
  // Obvious single-entry cases.
  if (PySlice_Check(obj)         // NOLINT
      || PyLong_Check(obj)       // NOLINT
      || obj == Py_Ellipsis      // NOLINT
      || obj == Py_None          // NOLINT
      || PyTensorCheck(obj)      // NOLINT
      || !PySequence_Check(obj)  // NOLINT
      || PyUnicode_Check(obj)) {
    tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(obj)));
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
      tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(obj)));
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
      tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(obj)));
      return tensor_index;
    }
    // Check whether we should unpack the index like a tuple.
    bool commit_to_unpack = false;
    for (Py_ssize_t i = 0; i < n; ++i) {
      PyObject* item = PySequence_GetItem(obj, i);
      if (commit_to_unpack) {
        CHECK_OR_RETURN(item) << "Sequence index is required.";
      } else {
        if (!item) {
          PyErr_Clear();
          break;
        }
        if (PySequence_Check(item)  // NOLINT
            || PySlice_Check(item)  // NOLINT
            || PyTensorCheck(item)  // NOLINT
            || item == Py_Ellipsis || item == Py_None) {
          commit_to_unpack = true;
        }
      }
      Py_DECREF(item);
    }
    if (commit_to_unpack) {
      tup = PySequence_Tuple(obj);
    } else {
      tensor_index->emplace_back(*JUST(detail::UnpackIndexItem(obj)));
      return tensor_index;
    }
  }

  tensor_index->resize(n);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_GetItem(tup, i);
    tensor_index->at(i) = *JUST(detail::UnpackIndexItem(item));
    Py_DECREF(item);
  }
  Py_DECREF(tup);
  return tensor_index;
}

// OpExpr
bool PyOpExprCheck(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::isinstance<OpExpr>(handle);
}

Maybe<OpExpr> PyUnpackOpExpr(PyObject* obj) {
  auto handle = py::reinterpret_borrow<py::object>(obj);
  return py::cast<std::shared_ptr<OpExpr>>(handle);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
