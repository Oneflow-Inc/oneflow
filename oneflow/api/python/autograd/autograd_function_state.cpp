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
#include "oneflow/api/python/autograd/autograd_function_state.h"

#include <pybind11/pybind11.h>
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/of_api_registry.h"

namespace py = pybind11;
namespace oneflow {
namespace one {
namespace {
inline FunctionAutoGradCaptureState* CheckAndGetStateData(PyAutogradFunctionState* state) {
  if (!state->data.lock()) {
    PyErr_Format(PyExc_RuntimeError, "Data is deallocated. Please don't hold context outside "
                                     "autograd.Function.forward or autograd.Function.backward");
    return nullptr;
  }
  return state->data.lock().get();
}
}  // namespace

#if PY_VERSION_HEX < 0x03070000
#define PYGETSET_NAME(name) const_cast<char*>(name)
#else
#define PYGETSET_NAME(name) (name)
#endif

#define PY_XINCREF(p) (({ Py_XINCREF(p); }), (p))

static PyObject* PyAutogradFunctionState_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyAutogradFunctionState* self = (PyAutogradFunctionState*)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->dynamic_attr_dict = PyDict_New();
    if (self->dynamic_attr_dict == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject*)self;
}

static void PyAutogradFunctionState_dealloc(PyAutogradFunctionState* self) {
  Py_XDECREF(self->dynamic_attr_dict);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// PyMethodDef start
static PyObject* PyAutogradFunctionState_save_for_backward(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  auto* _self = (PyAutogradFunctionState*)self;
  if (!functional::PyTensorSequenceCheck(args)) {
    return PyErr_Format(PyExc_TypeError, "save_for_backward() only support Tensor or Tensors");
  }
  const std::vector<std::shared_ptr<Tensor>>& tensor_list =
      functional::PyUnpackTensorSequence(args);
  for (const auto& tensor : tensor_list) {
    CheckAndGetStateData(_self)->SaveTensorForBackward(tensor);
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyObject* PyAutogradFunctionState_mark_non_differentiable(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  auto* _self = (PyAutogradFunctionState*)self;
  if (!functional::PyTensorSequenceCheck(args)) {
    return PyErr_Format(PyExc_TypeError, "save_for_backward() only support Tensor or Tensors");
  }
  const std::vector<std::shared_ptr<Tensor>>& tensor_list =
      functional::PyUnpackTensorSequence(args);
  for (const auto& tensor : tensor_list) {
    CheckAndGetStateData(_self)->MarkNonDifferentiable(tensor);
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyObject* PyAutogradFunctionState_is_data_valid(PyObject* self) {
  auto* _self = (PyAutogradFunctionState*)self;
  return functional::CastToPyObject(_self->data.lock() != nullptr);
}

static PyMethodDef PyAutogradFunctionState_methods[] = {
    {"save_for_backward", (PyCFunction)PyAutogradFunctionState_save_for_backward, METH_VARARGS,
     NULL},
    {"mark_non_differentiable", (PyCFunction)PyAutogradFunctionState_mark_non_differentiable,
     METH_VARARGS, NULL},
    {"_is_data_valid", (PyCFunction)PyAutogradFunctionState_is_data_valid, METH_NOARGS, NULL},
    {NULL} /* Sentinel */
};
// PyMethodDef end

// PyAutogradFunctionState_getset start
static PyObject* PyAutogradFunctionState_saved_tensors(PyObject* self, void*) {
  auto* _self = (PyAutogradFunctionState*)self;
  return functional::CastToPyObject<Maybe<TensorTuple>>(
      CheckAndGetStateData(_self)->SavedTensors());
}

static PyObject* PyAutogradFunctionState_get_dict(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  auto* _self = (PyAutogradFunctionState*)self;
  return _self->dynamic_attr_dict;
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyGetSetDef PyAutogradFunctionState_properties[] = {
    {PYGETSET_NAME("saved_tensors"), (getter)PyAutogradFunctionState_saved_tensors, NULL, NULL,
     NULL},
    {PYGETSET_NAME("__dict__"), (getter)PyAutogradFunctionState_get_dict, NULL, NULL, NULL},
    {NULL} /* Sentinel */
};
// PyAutogradFunctionState_getset end

PyObject* PyAutogradFunctionState_getattro(PyObject* self, PyObject* attr) {
  PyObject* res = NULL;
  res = PyDict_GetItem(((PyAutogradFunctionState*)self)->dynamic_attr_dict, attr);
  if (!res) {
    // Not found attr in dynamic_attr_dict, try to find it in tp_dict
    res = PyObject_GenericGetAttr(self, attr);
    if (!res) {
      return PyErr_Format(PyExc_AttributeError, "attribute %s not found", PyUnicode_AsUTF8(attr));
    }
  }
  return res;
}

int PyAutogradFunctionState_setattro(PyObject* self, PyObject* attr, PyObject* value) {
  auto* _self = (PyAutogradFunctionState*)self;
  return PyDict_SetItem(_self->dynamic_attr_dict, attr, value);
}

PyTypeObject PyAutogradFunctionState_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.autograd.Function.FunctionCtx", /* tp_name */
    sizeof(PyAutogradFunctionState),                                        /* tp_basicsize */
    0,                                                                      /* tp_itemsize */
    (destructor)PyAutogradFunctionState_dealloc,                            /* tp_dealloc */
    0,                                                    /* tp_vectorcall_offset */
    NULL,                                                 /* tp_getattr */
    NULL,                                                 /* tp_setattr */
    NULL,                                                 /* tp_reserved */
    NULL,                                                 /* tp_repr */
    NULL,                                                 /* tp_as_number */
    NULL,                                                 /* tp_as_sequence */
    NULL,                                                 /* tp_as_mapping */
    NULL,                                                 /* tp_hash  */
    NULL,                                                 /* tp_call */
    NULL,                                                 /* tp_str */
    PyAutogradFunctionState_getattro,                     /* tp_getattro */
    PyAutogradFunctionState_setattro,                     /* tp_setattro */
    NULL,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,             /* tp_flags */
    NULL,                                                 /* tp_doc */
    NULL,                                                 /* tp_traverse */
    NULL,                                                 /* tp_clear */
    NULL,                                                 /* tp_richcompare */
    0,                                                    /* tp_weaklistoffset */
    NULL,                                                 /* tp_iter */
    NULL,                                                 /* tp_iternext */
    PyAutogradFunctionState_methods,                      /* tp_methods */
    NULL,                                                 /* tp_members */
    PyAutogradFunctionState_properties,                   /* tp_getset */
    0,                                                    /* tp_base */
    NULL,                                                 /* tp_dict */
    NULL,                                                 /* tp_descr_get */
    NULL,                                                 /* tp_descr_set */
    offsetof(PyAutogradFunctionState, dynamic_attr_dict), /* tp_dictoffset */
    NULL,                                                 /* tp_init */
    NULL,                                                 /* tp_alloc */
    PyAutogradFunctionState_new,                          /* tp_new */
    NULL,                                                 /* tp_free */
};

PyObject* PyAutogradFunctionState_NewFromPtr(
    const std::shared_ptr<FunctionAutoGradCaptureState>& data) {
  if (!data) { Py_RETURN_NONE; }
  if (data->pyobject()) { return PY_XINCREF((PyObject*)data->pyobject()); }
  auto* self = (PyAutogradFunctionState*)(PyObject_CallObject(
      (PyObject*)&PyAutogradFunctionState_Type, NULL));
  if (self) {
    PY_XINCREF(self);
    self->data = data;
    CheckAndGetStateData(self)->set_pyobject_ptr(
        std::unique_ptr<void, void (*)(void*)>(self, [](void* ptr) { Py_DECREF((PyObject*)ptr); }));
  }
  return (PyObject*)self;
}

ONEFLOW_API_PYBIND11_MODULE("autograd.Function", m) {
  if (PyType_Ready(&PyAutogradFunctionState_Type) < 0) { return; }
  Py_INCREF(&PyAutogradFunctionState_Type);
  if (PyModule_AddObject(m.ptr(), "FunctionCtx", (PyObject*)&PyAutogradFunctionState_Type) < 0) {
    return;
  }
}

}  // namespace one
}  // namespace oneflow
