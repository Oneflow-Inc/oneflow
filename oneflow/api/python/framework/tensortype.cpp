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
#include <methodobject.h>
#include <object.h>
#include <pybind11/pybind11.h>
#include <functional>
#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/api/python/exception/exception.h"
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

PyObject* DTypeToTensortype(Symbol<DType> dtype) {
  if (dtype == DType::UInt8()) { return (PyObject*)PyByteTensortypeObject_Type; }
  if (dtype == DType::Int8()) { return (PyObject*)PyCharTensortypeObject_Type; }
  if (dtype == DType::Int16()) { return (PyObject*)PyShortTensortypeObject_Type; }
  if (dtype == DType::Int32()) { return (PyObject*)PyIntTensortypeObject_Type; }
  if (dtype == DType::Int64()) { return (PyObject*)PyLongTensortypeObject_Type; }
  if (dtype == DType::Float16()) { return (PyObject*)PyHalfTensortypeObject_Type; }
  if (dtype == DType::Float()) { return (PyObject*)PyFloatTensortypeObject_Type; }
  if (dtype == DType::Double()) { return (PyObject*)PyDoubleTensortypeObject_Type; }
  return PyErr_Format(PyExc_RuntimeError, "Invalid datatype");
}

// std::unordered_map<const Symbol<DType>&, PyObject*> dtype_to_tensortype_map {
//   {DType::UInt8(), (PyObject*)PyByteTensortypeObject_Type }  ,
//   {DType::Int8(),  (PyObject*)PyCharTensortypeObject_Type }  ,
//   {DType::Int16(), (PyObject*)PyShortTensortypeObject_Type } ,
//   {DType::Int32(), (PyObject*)PyIntTensortypeObject_Type }   ,
//   {DType::Int64(), (PyObject*)PyLongTensortypeObject_Type }  ,
//   {DType::Float16(), (PyObject*)PyHalfTensortypeObject_Type },
//   {DType::Float(), (PyObject*)PyFloatTensortypeObject_Type } ,
//   {DType::Double(), (PyObject*)PyDoubleTensortypeObject_Type },
// };

// std::unordered_map<PyObject*, const Symbol<DType>&> tensortype_to_dtype_map{
//     {(PyObject*)PyByteTensortypeObject_Type, DType::UInt8()},
//     {(PyObject*)PyCharTensortypeObject_Type, DType::Int8()},
//     {(PyObject*)PyShortTensortypeObject_Type, DType::Int16()},
//     {(PyObject*)PyIntTensortypeObject_Type, DType::Int32()},
//     {(PyObject*)PyLongTensortypeObject_Type, DType::Int64()},
//     {(PyObject*)PyHalfTensortypeObject_Type, DType::Float16()},
//     {(PyObject*)PyFloatTensortypeObject_Type, DType::Float()},
//     {(PyObject*)PyDoubleTensortypeObject_Type, DType::Double()},
// };

Symbol<DType> TensortypeToDType(PyObject* type) {
  if (type == (PyObject*)PyByteTensortypeObject_Type) return DType::UInt8();
  if (type == (PyObject*)PyCharTensortypeObject_Type) return DType::Int8();
  if (type == (PyObject*)PyShortTensortypeObject_Type) return DType::Int16();
  if (type == (PyObject*)PyIntTensortypeObject_Type) return DType::Int32();
  if (type == (PyObject*)PyLongTensortypeObject_Type) return DType::Int64();
  if (type == (PyObject*)PyHalfTensortypeObject_Type) return DType::Float16();
  if (type == (PyObject*)PyFloatTensortypeObject_Type) return DType::Float();
  if (type == (PyObject*)PyDoubleTensortypeObject_Type) return DType::Double();
  CHECK_OR_THROW(false) << "invalid tensortype";
  return DType::UInt8();
}

static PyObject* TensortypeMetaCls_call(PyObject* type, PyObject* args, PyObject* kwargs) {
  return ((PyTypeObject*)type)->tp_call(type, args, kwargs);
}

static void TensortypeMetaCls_dealloc(PyObject* type) { PyType_Type.tp_dealloc(type); }

// tensortype
static PyHeapTypeObject* MakeTensortypeMetaclass() {
  PyObject* name = PyUnicode_FromString("tensortype");

  auto* heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);

  auto* type = &heap_type->ht_type;
  // auto* type = (PyTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);

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

static PyObject* test_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* tensor = NULL;
  static const char* keywords[2] = {"tensor", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(keywords),
                                   &tensor)) {
    return NULL;
  }
  if (!PyTensor_Check(tensor)) {
    return NULL;
  }

  Symbol<DType> dtype = TensortypeToDType(self);
  const auto& t = PyTensor_Unpack(tensor);
  const auto& cast_t = functional::To(t, dtype, false);
  return functional::CastToPyObject(cast_t);
  END_HANDLE_ERRORS
}

static PyTypeObject* MakeTensortypeType(const char* tensortype_name, pybind11::module_& m) {
  PyObject* name = PyUnicode_FromString(tensortype_name);

  auto* metaclass = &TensortypeMetaclass_Type->ht_type;
  auto* heap_type = (PyHeapTypeObject*)metaclass->tp_alloc(metaclass, 0);
  if (!heap_type) { return NULL; }
  heap_type->ht_qualname = PY_XINCREF(name);
  auto* type = &heap_type->ht_type;
  type->tp_base = (PyTypeObject*)TensortypeMetaclass_Type;
  type->tp_name = tensortype_name;
  type->tp_basicsize = sizeof(PyTensorObject);
  type->tp_call = test_call;
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow"));
  if (type && PyModule_AddObject(m.ptr(), tensortype_name, (PyObject*)type) < 0) { return NULL; }
  return type;
}

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  PyByteTensortypeObject_Type = MakeTensortypeType("ByteTensor", m);
  PyCharTensortypeObject_Type = MakeTensortypeType("CharTensor", m);
  PyShortTensortypeObject_Type = MakeTensortypeType("ShortTensor", m);
  PyIntTensortypeObject_Type = MakeTensortypeType("IntTensor", m);
  PyLongTensortypeObject_Type = MakeTensortypeType("LongTensor", m);
  PyHalfTensortypeObject_Type = MakeTensortypeType("HalfTensor", m);
  PyFloatTensortypeObject_Type = MakeTensortypeType("FloatTensor", m);
  PyDoubleTensortypeObject_Type = MakeTensortypeType("DoubleTensor", m);
}