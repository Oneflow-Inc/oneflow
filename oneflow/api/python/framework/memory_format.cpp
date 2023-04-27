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
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/functional/common.h"

#include "oneflow/api/python/framework/memory_format.h"

namespace py = pybind11;

namespace oneflow {

static PyObject* PyMemoryFormat_repr(PyMemoryFormatObject* self) {
  auto memory_format = PyMemoryFormat_Unpack((PyObject*)self);
  if (memory_format == MemoryFormat::kContiguous) {
    return PyUnicode_FromString("oneflow.contiguous_format");
  } else if (memory_format == MemoryFormat::kChannelsLast) {
    return PyUnicode_FromString("oneflow.channels_last");
  } else if (memory_format == MemoryFormat::kPreserve) {
    return PyUnicode_FromString("oneflow.preserve_format");
  } else {
    THROW(TypeError) << "invalid memory format";
    return nullptr;
  }
}

PyTypeObject PyMemoryFormat_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.memory_format", /* tp_name */
    sizeof(PyMemoryFormatObject),                           /* tp_basicsize */
    0,                                                      /* tp_itemsize */
    NULL,                                                   /* tp_dealloc */
    0,                                                      /* tp_vectorcall_offset */
    NULL,                                                   /* tp_getattr */
    NULL,                                                   /* tp_setattr */
    NULL,                                                   /* tp_reserved */
    (reprfunc)PyMemoryFormat_repr,                          /* tp_repr */
    NULL,                                                   /* tp_as_number */
    NULL,                                                   /* tp_as_sequence */
    NULL,                                                   /* tp_as_mapping */
    NULL,                                                   /* tp_hash  */
    NULL,                                                   /* tp_call */
    NULL,                                                   /* tp_str */
    NULL,                                                   /* tp_getattro */
    NULL,                                                   /* tp_setattro */
    NULL,                                                   /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,               /* tp_flags */
};

bool PyMemoryFormat_Check(PyObject* self) { return self && self->ob_type == &PyMemoryFormat_Type; }

PyObject* PyMemoryFormat_New(MemoryFormat memory_format) {
  auto* self = (PyMemoryFormatObject*)PyMemoryFormat_Type.tp_alloc(&PyMemoryFormat_Type, 0);
  self->memory_format = memory_format;
  return (PyObject*)self;
}

static PyObject* PyMemoryFormat_contiguous = nullptr;
static PyObject* PyMemoryFormat_channels_last = nullptr;
static PyObject* PyMemoryFormat_preserve = nullptr;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  if (PyType_Ready(&PyMemoryFormat_Type) < 0) { return; }
  Py_INCREF(&PyMemoryFormat_Type);
  if (PyModule_AddObject(m.ptr(), "memory_format", (PyObject*)&PyMemoryFormat_Type) < 0) { return; }

  PyMemoryFormat_contiguous = PyMemoryFormat_New(MemoryFormat::kContiguous);
  PyMemoryFormat_channels_last = PyMemoryFormat_New(MemoryFormat::kChannelsLast);
  PyMemoryFormat_preserve = PyMemoryFormat_New(MemoryFormat::kPreserve);
  if (PyModule_AddObject(m.ptr(), "contiguous_format", PyMemoryFormat_contiguous) < 0) { return; }
  if (PyModule_AddObject(m.ptr(), "channels_last", PyMemoryFormat_channels_last) < 0) { return; }
  if (PyModule_AddObject(m.ptr(), "preserve_format", PyMemoryFormat_preserve) < 0) { return; }
}

}  // namespace oneflow
