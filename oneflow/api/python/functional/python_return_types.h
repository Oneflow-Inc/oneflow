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

// This code is referenced from:
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/structseq.cpp

#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_RETURN_TYPES_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_RETURN_TYPES_H_

#include <Python.h>
#undef _PyGC_FINALIZED
#include <string>
#include <sstream>
#include <structmember.h>

#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"

namespace oneflow {
namespace one {
namespace functional {

inline PyObject* toTuple(PyStructSequence* obj) {
#if PY_MAJOR_VERSION == 2
  ROF_RUNTIME_ERROR() << "Oneflow do not support python 2";
#else
  Py_INCREF(obj);
  return (PyObject*)obj;
#endif
}

PyObject* returned_structseq_repr(PyStructSequence* obj) {
  HANDLE_ERRORS
  PyTypeObject* tp = Py_TYPE(obj);
  PyObject* tuple = toTuple(obj);
  if (tuple == nullptr) { return nullptr; }

  std::stringstream ss;
  ss << tp->tp_name << "(\n";
  Py_ssize_t num_elements = Py_SIZE(obj);

  for (Py_ssize_t i = 0; i < num_elements; i++) {
    const char* cname = tp->tp_members[i].name;
    if (cname == nullptr) {
      PyErr_Format(PyExc_SystemError,
                   "In structseq_repr(), member %zd name is nullptr"
                   " for type %.500s",
                   i, tp->tp_name);
      Py_DECREF(tuple);
      return nullptr;
    }

    PyObject* val = PyTuple_GetItem(tuple, i);
    if (val == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }

    auto repr = PyObject_Repr(val);
    if (repr == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }

    const char* crepr = PyUnicode_AsUTF8(repr);
    Py_DECREF(repr);
    if (crepr == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }

    ss << cname << '=' << crepr;
    if (i < num_elements - 1) { ss << ",\n"; }
  }
  ss << ")";

  Py_DECREF(tuple);
  return PyUnicode_FromString(ss.str().c_str());
  END_HANDLE_ERRORS
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_RETURN_TYPES_H_
