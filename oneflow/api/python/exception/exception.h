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
#ifndef ONEFLOW_API_PYTHON_COMMON_EXCEPTION_H_
#define ONEFLOW_API_PYTHON_COMMON_EXCEPTION_H_

#include <Python.h>
#include <pybind11/pybind11.h>

#include "oneflow/core/common/exception.h"

namespace py = pybind11;

#define HANDLE_ERRORS try {
#define END_HANDLE_ERRORS_RETSTMT(retstmt)                \
  }                                                       \
  catch (py::error_already_set & e) {                     \
    e.restore();                                          \
    retstmt;                                              \
  }                                                       \
  catch (const oneflow::RuntimeException& e) {            \
    PyErr_SetString(PyExc_RuntimeError, e.what());        \
    retstmt;                                              \
  }                                                       \
  catch (const oneflow::IndexException& e) {              \
    PyErr_SetString(PyExc_IndexError, e.what());          \
    retstmt;                                              \
  }                                                       \
  catch (const oneflow::TypeException& e) {               \
    PyErr_SetString(PyExc_TypeError, e.what());           \
    retstmt;                                              \
  }                                                       \
  catch (const oneflow::NotImplementedException& e) {     \
    PyErr_SetString(PyExc_NotImplementedError, e.what()); \
    retstmt;                                              \
  }                                                       \
  catch (const std::exception& e) {                       \
    PyErr_SetString(PyExc_RuntimeError, e.what());        \
    retstmt;                                              \
  }

#define END_HANDLE_ERRORS END_HANDLE_ERRORS_RETSTMT(return NULL)
#define END_HANDLE_ERRORS_RET(retval) END_HANDLE_ERRORS_RETSTMT(return retval)
#define END_HANDLE_ERRORS_NORET END_HANDLE_ERRORS_RETSTMT(void)

#endif  // ONEFLOW_API_PYTHON_COMMON_EXCEPTION_H_
