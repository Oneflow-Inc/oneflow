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
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/shape.h"

namespace py = pybind11;

namespace oneflow {

using one::functional::PyObjectPtr;

static PyObject* TensorSize_repr(TensorSize* self) {
  std::stringstream ss;
  int32_t idx = 0;
  int32_t size = PyTuple_Size((PyObject*)self);
  ss << "oneflow.Size([";
  for (int i = 0; i < size; ++i) {
    int64_t dim = PyLong_AsLongLong(PyTuple_GET_ITEM(self, i));
    ss << dim;
    if (++idx != size) { ss << ", "; }
  }
  ss << "])";
  return PyUnicode_FromString(ss.str().c_str());
}

static PyObject* TensorSize_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObjectPtr self(PyTuple_Type.tp_new(type, args, kwargs));
  if (self.get()) {
    for (int i = 0; i < PyTuple_Size(self.get()); ++i) {
      PyObject* item = PyTuple_GET_ITEM(self.get(), i);
      if (!PyLong_Check(item)) {
        return PyErr_Format(PyExc_TypeError,
                            "oneflow.Size() takes an iterable of 'int', but item '%d' is '%s'", i,
                            Py_TYPE(item)->tp_name);
      }
    }
  }
  return self.release();
}

static Py_ssize_t TensorSize_length(TensorSize* self) {
  return PyTuple_Type.tp_as_sequence->sq_length((PyObject*)self);
}

static PyObject* TensorSize_concat(TensorSize* self, PyObject* other) {
  PyObjectPtr result(PyTuple_Type.tp_as_sequence->sq_concat((PyObject*)self, other));
  if (!result.get()) { return nullptr; }
  if (PyTuple_Check(result.get())) {
    PyObjectPtr args(PyTuple_Pack(1, result.get()));
    return TensorSize_new(&TensorSize_Type, args.get(), nullptr);
  }
  return result.release();
}

static PyObject* TensorSize_repeat(TensorSize* self, Py_ssize_t n) {
  PyObjectPtr result(PyTuple_Type.tp_as_sequence->sq_repeat((PyObject*)self, n));
  if (!result.get()) { return nullptr; }
  if (PyTuple_Check(result.get())) {
    PyObjectPtr args(PyTuple_Pack(1, result.get()));
    return TensorSize_new(&TensorSize_Type, args.get(), nullptr);
  }
  return result.release();
}

static PyObject* TensorSize_item(TensorSize* self, Py_ssize_t i) {
  return PyTuple_Type.tp_as_sequence->sq_item((PyObject*)self, i);
}

static int TensorSize_contains(TensorSize* self, PyObject* el) {
  return PyTuple_Type.tp_as_sequence->sq_contains((PyObject*)self, el);
}

static PySequenceMethods TensorSize_as_sequence = {
    (lenfunc)TensorSize_length,      /* sq_length */
    (binaryfunc)TensorSize_concat,   /* sq_concat */
    (ssizeargfunc)TensorSize_repeat, /* sq_repeat */
    (ssizeargfunc)TensorSize_item,   /* sq_item */
    0,                               /* sq_slice */
    0,                               /* sq_ass_item */
    0,                               /* sq_ass_slice */
    (objobjproc)TensorSize_contains, /* sq_contains */
};

static PyObject* TensorSize_subscript(TensorSize* self, PyObject* item) {
  PyObjectPtr result(PyTuple_Type.tp_as_mapping->mp_subscript((PyObject*)self, item));
  if (!result.get()) { return nullptr; }
  if (PyTuple_Check(result.get())) {
    PyObjectPtr args(PyTuple_Pack(1, result.get()));
    return TensorSize_new(&TensorSize_Type, args.get(), nullptr);
  }
  return result.release();
};

static PyMappingMethods TensorSize_as_mapping = {
    (lenfunc)TensorSize_length,       /* mp_length */
    (binaryfunc)TensorSize_subscript, /* mp_subscript */
    0,                                /* mp_ass_subscript */
};

static PyObject* TensorSize_numel(PyObject* self, PyObject* args) {
  int64_t numel = 1;
  for (int i = 0; i < PyTuple_Size(self); ++i) {
    numel *= PyLong_AsLongLong(PyTuple_GET_ITEM((TensorSize*)self, i));
  }
  return PyLong_FromLongLong(numel);
}

static PyMethodDef TensorSize_methods[] = {
    {"numel", (PyCFunction)TensorSize_numel, METH_NOARGS, NULL}, {NULL}};

PyTypeObject TensorSize_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.Size", /* tp_name */
    sizeof(TensorSize),                            /* tp_basicsize */
    0,                                             /* tp_itemsize */
    NULL,                                          /* tp_dealloc */
    0,                                             /* tp_vectorcall_offset */
    NULL,                                          /* tp_getattr */
    NULL,                                          /* tp_setattr */
    NULL,                                          /* tp_reserved */
    (reprfunc)TensorSize_repr,                     /* tp_repr */
    NULL,                                          /* tp_as_number */
    &TensorSize_as_sequence,                       /* tp_as_sequence */
    &TensorSize_as_mapping,                        /* tp_as_mapping */
    NULL,                                          /* tp_hash  */
    NULL,                                          /* tp_call */
    NULL,                                          /* tp_str */
    NULL,                                          /* tp_getattro */
    NULL,                                          /* tp_setattro */
    NULL,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,      /* tp_flags */
    NULL,                                          /* tp_doc */
    NULL,                                          /* tp_traverse */
    NULL,                                          /* tp_clear */
    NULL,                                          /* tp_richcompare */
    0,                                             /* tp_weaklistoffset */
    NULL,                                          /* tp_iter */
    NULL,                                          /* tp_iternext */
    TensorSize_methods,                            /* tp_methods */
    NULL,                                          /* tp_members */
    NULL,                                          /* tp_getset */
    &PyTuple_Type,                                 /* tp_base */
    NULL,                                          /* tp_dict */
    NULL,                                          /* tp_descr_get */
    NULL,                                          /* tp_descr_set */
    0,                                             /* tp_dictoffset */
    NULL,                                          /* tp_init */
    NULL,                                          /* tp_alloc */
    TensorSize_new,                                /* tp_new */
    NULL,                                          /* tp_free */
};

int TensorSize_Check(PyObject* p) { return p && p->ob_type == &TensorSize_Type; }

PyObject* TensorSize_New(Py_ssize_t len) { return TensorSize_Type.tp_alloc(&TensorSize_Type, len); }

PyObject* TensorSize_NewFromShape(const Shape& size) {
  PyObjectPtr self(TensorSize_New(size.NumAxes()));
  if (self.get()) {
    for (int i = 0; i < size.NumAxes(); ++i) {
      PyTuple_SET_ITEM(self.get(), i, PyLong_FromLongLong(size.At(i)));
    }
  }
  return self.release();
}

Shape TensorSize_AsShape(PyObject* self) {
  if (!TensorSize_Check(self)) {
    PyErr_Format(PyExc_TypeError, "can only convert TensorSize(not \"%s\") to Shape",
                 Py_TYPE(self)->tp_name);
    return Shape();
  }
  int size = TensorSize_length((TensorSize*)self);
  DimVector dim_vec(size);
  for (int i = 0; i < size; ++i) {
    dim_vec[i] = PyLong_AsLongLong(PyTuple_GET_ITEM((TensorSize*)self, i));
  }
  return Shape(std::move(dim_vec));
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  if (PyType_Ready(&TensorSize_Type) < 0) { return; }
  Py_INCREF(&TensorSize_Type);
  if (PyModule_AddObject(m.ptr(), "Size", (PyObject*)&TensorSize_Type) < 0) { return; }
}

}  // namespace oneflow
