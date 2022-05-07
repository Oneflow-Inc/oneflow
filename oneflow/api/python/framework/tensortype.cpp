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
#include <objimpl.h>
#include <pybind11/pybind11.h>
#include <strings.h>
#include <functional>
#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/api/python/framework/tensortype.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/api/python/exception/exception.h"
namespace py = pybind11;

namespace oneflow {
namespace one {
#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()
#define PY_XINCREF(p) (({ Py_XINCREF(p); }), (p))

PyHeapTypeObject* PyTensortypeMetaclass_Type = NULL;
PyTypeObject* PyByteTensortypeObject_Type = NULL;
PyTypeObject* PyCharTensortypeObject_Type = NULL;
PyTypeObject* PyShortTensortypeObject_Type = NULL;
PyTypeObject* PyIntTensortypeObject_Type = NULL;
PyTypeObject* PyLongTensortypeObject_Type = NULL;
PyTypeObject* PyHalfTensortypeObject_Type = NULL;
PyTypeObject* PyFloatTensortypeObject_Type = NULL;
PyTypeObject* PyDoubleTensortypeObject_Type = NULL;

PyHeapTypeObject* PyCudaTensortypeMetaclass_Type = NULL;
PyTypeObject* PyCudaByteTensortypeObject_Type = NULL;
PyTypeObject* PyCudaCharTensortypeObject_Type = NULL;
PyTypeObject* PyCudaShortTensortypeObject_Type = NULL;
PyTypeObject* PyCudaIntTensortypeObject_Type = NULL;
PyTypeObject* PyCudaLongTensortypeObject_Type = NULL;
PyTypeObject* PyCudaHalfTensortypeObject_Type = NULL;
PyTypeObject* PyCudaFloatTensortypeObject_Type = NULL;
PyTypeObject* PyCudaDoubleTensortypeObject_Type = NULL;

PyObject* GetTensortype(const Symbol<DType>& dtype, const Maybe<Symbol<Device>>& device) {
  auto devicetype = CHECK_JUST(device)->enum_type();
  if (devicetype == DeviceType::kCPU) {
    if (dtype == DType::UInt8()) { return (PyObject*)PyByteTensortypeObject_Type; }
    if (dtype == DType::Int8()) { return (PyObject*)PyCharTensortypeObject_Type; }
    if (dtype == DType::Int16()) { return (PyObject*)PyShortTensortypeObject_Type; }
    if (dtype == DType::Int32()) { return (PyObject*)PyIntTensortypeObject_Type; }
    if (dtype == DType::Int64()) { return (PyObject*)PyLongTensortypeObject_Type; }
    if (dtype == DType::Float16()) { return (PyObject*)PyHalfTensortypeObject_Type; }
    if (dtype == DType::Float()) { return (PyObject*)PyFloatTensortypeObject_Type; }
    if (dtype == DType::Double()) { return (PyObject*)PyDoubleTensortypeObject_Type; }
    return PyErr_Format(PyExc_RuntimeError, "Invalid datatype");
  } else if (devicetype == DeviceType::kCUDA) {
    if (dtype == DType::UInt8()) { return (PyObject*)PyCudaByteTensortypeObject_Type; }
    if (dtype == DType::Int8()) { return (PyObject*)PyCudaCharTensortypeObject_Type; }
    if (dtype == DType::Int16()) { return (PyObject*)PyCudaShortTensortypeObject_Type; }
    if (dtype == DType::Int32()) { return (PyObject*)PyCudaIntTensortypeObject_Type; }
    if (dtype == DType::Int64()) { return (PyObject*)PyCudaLongTensortypeObject_Type; }
    if (dtype == DType::Float16()) { return (PyObject*)PyCudaHalfTensortypeObject_Type; }
    if (dtype == DType::Float()) { return (PyObject*)PyCudaFloatTensortypeObject_Type; }
    if (dtype == DType::Double()) { return (PyObject*)PyCudaDoubleTensortypeObject_Type; }
    return PyErr_Format(PyExc_RuntimeError, "Invalid datatype");
  }
  return PyErr_Format(PyExc_RuntimeError, "Invalid device");
}

DeviceType TensortypeToDevice(PyObject* tensortype) {
  if (((PyTypeObject*)tensortype)->tp_base == (PyTypeObject*)PyTensortypeMetaclass_Type)
    return DeviceType::kCPU;
  else
    return DeviceType::kCUDA;
}

Symbol<DType> TensortypeToDType(PyObject* type) {
  bool is_tensortype =
      ((PyHeapTypeObject*)type)->ht_type.tp_base == (PyTypeObject*)PyTensortypeMetaclass_Type
      || ((PyHeapTypeObject*)type)->ht_type.tp_base
             == (PyTypeObject*)PyCudaTensortypeMetaclass_Type;
  CHECK_OR_THROW(is_tensortype) << "invalid tensortype";
  if (type == (PyObject*)PyByteTensortypeObject_Type) return DType::UInt8();
  if (type == (PyObject*)PyCharTensortypeObject_Type) return DType::Int8();
  if (type == (PyObject*)PyShortTensortypeObject_Type) return DType::Int16();
  if (type == (PyObject*)PyIntTensortypeObject_Type) return DType::Int32();
  if (type == (PyObject*)PyLongTensortypeObject_Type) return DType::Int64();
  if (type == (PyObject*)PyHalfTensortypeObject_Type) return DType::Float16();
  if (type == (PyObject*)PyFloatTensortypeObject_Type) return DType::Float();
  if (type == (PyObject*)PyDoubleTensortypeObject_Type) return DType::Double();

  if (type == (PyObject*)PyCudaByteTensortypeObject_Type) return DType::UInt8();
  if (type == (PyObject*)PyCudaCharTensortypeObject_Type) return DType::Int8();
  if (type == (PyObject*)PyCudaShortTensortypeObject_Type) return DType::Int16();
  if (type == (PyObject*)PyCudaIntTensortypeObject_Type) return DType::Int32();
  if (type == (PyObject*)PyCudaLongTensortypeObject_Type) return DType::Int64();
  if (type == (PyObject*)PyCudaHalfTensortypeObject_Type) return DType::Float16();
  if (type == (PyObject*)PyCudaFloatTensortypeObject_Type) return DType::Float();
  if (type == (PyObject*)PyCudaDoubleTensortypeObject_Type) return DType::Double();
  return DType::UInt8();
}


static PyObject* TensortypeType_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* tensor = NULL;
  static const char* keywords[2] = {"tensor", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(keywords), &tensor)) {
    return NULL;
  }
  if (!PyTensor_Check(tensor)) { return NULL; }

  Symbol<oneflow::DType> dtype = TensortypeToDType(self);
  Optional<std::string> device_str = "cuda";
  if (TensortypeToDevice(self) == DeviceType::kCPU) device_str = "cpu";

  const auto& t = PyTensor_Unpack(tensor);
  const auto& cast_t = functional::To(t, device_str, dtype, false);
  return functional::CastToPyObject(cast_t);
  END_HANDLE_ERRORS
}

// tensortype
static PyHeapTypeObject* MakeTensortypeMetaclass() {
  PyObject* name = PyUnicode_FromString("tensortype");

  auto* heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);

  auto* type = &heap_type->ht_type;

  type->tp_name = "tensortype";
  type->tp_base = PY_XINCREF(&PyType_Type);
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  type->tp_call = TensortypeType_call;
  type->tp_dealloc = PyType_Type.tp_dealloc;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow"));
  return heap_type;
}

static PyTypeObject* MakeTensortypeType(const char* tensortype_name, const DeviceType device,
                                        pybind11::module_& m) {
  PyObject* name = PyUnicode_FromString(tensortype_name);

  auto* metaclass = &PyTensortypeMetaclass_Type->ht_type;
  auto* heap_type = (PyHeapTypeObject*)metaclass->tp_alloc(metaclass, 0);
  if (!heap_type) { return NULL; }
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);
  auto* type = &heap_type->ht_type;
  type->tp_base = (PyTypeObject*)PyTensortypeMetaclass_Type;
  type->tp_name = tensortype_name;
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow"));
  if (device == DeviceType::kCUDA) {
    PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow.cuda"));
    auto cuda = m.def_submodule("cuda");
    type->tp_base = (PyTypeObject*)PyCudaTensortypeMetaclass_Type;
    ((PyObject*)type)->ob_type = &PyCudaTensortypeMetaclass_Type->ht_type;
    if (type && PyModule_AddObject(cuda.ptr(), tensortype_name, (PyObject*)type) < 0) {
      return NULL;
    }
  } else {
    ((PyObject*)type)->ob_type = &PyTensortypeMetaclass_Type->ht_type;
    if (type && PyModule_AddObject(m.ptr(), tensortype_name, (PyObject*)type) < 0) { return NULL; }
  }
  return type;
}

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  PyTensortypeMetaclass_Type = MakeTensortypeMetaclass();
  PyByteTensortypeObject_Type = MakeTensortypeType("ByteTensor", oneflow::DeviceType::kCPU, m);
  PyCharTensortypeObject_Type = MakeTensortypeType("CharTensor", oneflow::DeviceType::kCPU, m);
  PyShortTensortypeObject_Type = MakeTensortypeType("ShortTensor", oneflow::DeviceType::kCPU, m);
  PyIntTensortypeObject_Type = MakeTensortypeType("IntTensor", oneflow::DeviceType::kCPU, m);
  PyLongTensortypeObject_Type = MakeTensortypeType("LongTensor", oneflow::DeviceType::kCPU, m);
  PyHalfTensortypeObject_Type = MakeTensortypeType("HalfTensor", oneflow::DeviceType::kCPU, m);
  PyFloatTensortypeObject_Type = MakeTensortypeType("FloatTensor", oneflow::DeviceType::kCPU, m);
  PyDoubleTensortypeObject_Type = MakeTensortypeType("DoubleTensor", oneflow::DeviceType::kCPU, m);

  PyCudaTensortypeMetaclass_Type = MakeTensortypeMetaclass();
  PyCudaByteTensortypeObject_Type = MakeTensortypeType("ByteTensor", oneflow::DeviceType::kCUDA, m);
  PyCudaCharTensortypeObject_Type = MakeTensortypeType("CharTensor", oneflow::DeviceType::kCUDA, m);
  PyCudaShortTensortypeObject_Type =
      MakeTensortypeType("ShortTensor", oneflow::DeviceType::kCUDA, m);
  PyCudaIntTensortypeObject_Type = MakeTensortypeType("IntTensor", oneflow::DeviceType::kCUDA, m);
  PyCudaLongTensortypeObject_Type = MakeTensortypeType("LongTensor", oneflow::DeviceType::kCUDA, m);
  PyCudaHalfTensortypeObject_Type = MakeTensortypeType("HalfTensor", oneflow::DeviceType::kCUDA, m);
  PyCudaFloatTensortypeObject_Type =
      MakeTensortypeType("FloatTensor", oneflow::DeviceType::kCUDA, m);
  PyCudaDoubleTensortypeObject_Type =
      MakeTensortypeType("DoubleTensor", oneflow::DeviceType::kCUDA, m);
}