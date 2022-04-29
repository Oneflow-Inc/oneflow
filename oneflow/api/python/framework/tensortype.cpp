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
#include <Python.h>
#include <object.h>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/dtype.h"
namespace py = pybind11;

namespace oneflow {
namespace one {
#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()
#define PY_XINCREF(p) (({ Py_XINCREF(p); }), (p))

PyTypeObject* PyTensortypeObject_Type = NULL;

PyTypeObject* PyByteTensortypeObject_Type = NULL;
PyTypeObject* PyCharTensortypeObject_Type = NULL;
PyTypeObject* PyShortTensortypeObject_Type = NULL;
PyTypeObject* PyIntTensortypeObject_Type = NULL;
PyTypeObject* PyLongTensortypeObject_Type = NULL;

PyTypeObject* PyHalfTensortypeObject_Type = NULL;
PyTypeObject* PyFloatTensortypeObject_Type = NULL;
PyTypeObject* PyDoubleTensortypeObject_Type = NULL;

static PyObject* DTypeToTensortype(Symbol<DType> dtype) {
  if (dtype == DType::UInt8()) { return (PyObject*)PyByteTensortypeObject_Type; }
  if (dtype == DType::Int8()) { return (PyObject*)PyCharTensortypeObject_Type; }
  if (dtype == DType::Int16()) { return (PyObject*)PyShortTensortypeObject_Type; }
  if (dtype == DType::Int32()) { return (PyObject*)PyIntTensortypeObject_Type; }
  if (dtype == DType::Int64()) { return (PyObject*)PyLongTensortypeObject_Type; }

  if (dtype == DType::Float16()) { return (PyObject*)PyHalfTensortypeObject_Type; }
  if (dtype == DType::Float()) { return (PyObject*)PyFloatTensortypeObject_Type; }
  if (dtype == DType::Double()) { return (PyObject*)PyDoubleTensortypeObject_Type; }
  return NULL;
}

static PyObject* TensortypeMetaCls_call(PyObject* type, PyObject* args, PyObject* kwargs) {
  return PyType_Type.tp_call(type, args, kwargs);
}

static void TensortypeMetaCls_dealloc(PyObject* type) { PyType_Type.tp_dealloc(type); }

static PyHeapTypeObject* MakeTensortypeMetaclass() {
  PyObject* name = PyUnicode_FromString("tensortype");

  auto* heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);

  auto* type = &heap_type->ht_type;
  type->tp_name = "tensortype";
  type->tp_base = PY_XINCREF(&PyType_Type);
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  type->tp_call = TensortypeMetaCls_call;
  type->tp_dealloc = TensortypeMetaCls_dealloc;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow"));
  return heap_type;
}

static PyHeapTypeObject* TensortypeMetaclass_Type = MakeTensortypeMetaclass();

static PyTypeObject* MakeTensortypeType(const char* tensortype_name) {
  PyObject* name = PyUnicode_FromString(tensortype_name);

  auto* metaclass = &TensortypeMetaclass_Type->ht_type;
  auto* heap_type = (PyHeapTypeObject*)metaclass->tp_alloc(metaclass, 0);
  if (!heap_type) { return NULL; }
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);
  auto* type = &heap_type->ht_type;
  type->tp_name = "Tensor";
  type->tp_basicsize = sizeof(PyTensorObject);

  // type->tp_init = PyTensorObject_init;
  // type->tp_dealloc = PyTensorObject_dealloc;
  // type->tp_getset = PyTensorObject_properties;
  // type->tp_methods = PyTensorObject_methods;

  type->tp_as_number = &heap_type->as_number;
  // type->tp_as_sequence = &PyTensorObject_as_sequence;
  // type->tp_as_mapping = &PyTensorObject_as_mapping;

  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow"));
  return type;
}

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  PyByteTensortypeObject_Type = MakeTensortypeType("ByteTensor");
  if (PyByteTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "ByteTensor", (PyObject*)PyByteTensortypeObject_Type) < 0) {
    return;
  }
  PyCharTensortypeObject_Type = MakeTensortypeType("CharTensor");
  if (PyCharTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "CharTensor", (PyObject*)PyCharTensortypeObject_Type) < 0) {
    return;
  }
  PyShortTensortypeObject_Type = MakeTensortypeType("ShortTensor");
  if (PyShortTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "ShortTensor", (PyObject*)PyShortTensortypeObject_Type) < 0) {
    return;
  }
  PyIntTensortypeObject_Type = MakeTensortypeType("IntTensor");
  if (PyIntTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "IntTensor", (PyObject*)PyIntTensortypeObject_Type) < 0) {
    return;
  }
  PyLongTensortypeObject_Type = MakeTensortypeType("LongTensor");
  if (PyLongTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "LongTensor", (PyObject*)PyLongTensortypeObject_Type) < 0) {
    return;
  }
  PyHalfTensortypeObject_Type = MakeTensortypeType("HalfTensor");
  if (PyHalfTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "HalfTensor", (PyObject*)PyHalfTensortypeObject_Type) < 0) {
    return;
  }
  PyFloatTensortypeObject_Type = MakeTensortypeType("FloatTensor");
  if (PyFloatTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "FloatTensor", (PyObject*)PyFloatTensortypeObject_Type) < 0) {
    return;
  }
  PyDoubleTensortypeObject_Type = MakeTensortypeType("DoubleTensor");
  if (PyDoubleTensortypeObject_Type
      && PyModule_AddObject(m.ptr(), "DoubleTensor", (PyObject*)PyDoubleTensortypeObject_Type)
             < 0) {
    return;
  }
}