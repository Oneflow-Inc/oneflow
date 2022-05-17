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
#include "oneflow/core/common/throw.h"

namespace py = pybind11;

namespace oneflow {

py::object AddFunctionDoc(py::object f, const std::string& doc_string) {
  static std::vector<std::string> all_doc_strings;
  all_doc_strings.emplace_back(doc_string);
  const char* doc_str = all_doc_strings.back().c_str();
  PyObject* obj = f.ptr();
  if (PyCFunction_Check(obj)) {
    auto* f = (PyCFunctionObject*)obj;
    if (f->m_ml->ml_doc) {
      THROW(RuntimeError) << "function " << f->m_ml->ml_name << " already has a docstring "
                          << "shows: " << f->m_ml->ml_doc;
    }
    f->m_ml->ml_doc = doc_str;
  } else if (PyFunction_Check(obj)) {
    auto* f = (PyFunctionObject*)obj;
    if (f->func_doc != Py_None) {
      THROW(RuntimeError) << "function "
                          << PyBytes_AsString(
                                 PyUnicode_AsEncodedString(f->func_name, "utf-8", "~E~"))
                          << " already has a docstring";
    }
    f->func_doc = PyUnicode_FromString(doc_str);
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* f = (PyMethodDescrObject*)obj;
    if (f->d_method->ml_doc) {
      THROW(RuntimeError) << "function " << f->d_method->ml_name << "already has a docstring";
    }
    f->d_method->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    PyMethodDescrObject* f = (PyMethodDescrObject*)obj;
    if (f->d_method->ml_doc) {
      THROW(RuntimeError) << "function " << f->d_method->ml_name << "already has a docstring";
    }
    f->d_method->ml_doc = doc_str;
  } else if (py::isinstance<py::detail::generic_type>(f)) {
    if (py::hasattr(f, "__doc__")) {
      auto doc = py::getattr(f, "__doc__");
      if (!doc.is(py::none())) {
        THROW(RuntimeError) << Py_TYPE(obj)->tp_name << " already has a docstring";
      }
    }
    py::setattr(f, "__doc__", py::reinterpret_steal<py::object>(PyUnicode_FromString(doc_str)));
  } else if (Py_TYPE(obj)->tp_name == PyProperty_Type.tp_name) {
    py::setattr(f, "__doc__", py::reinterpret_steal<py::object>(PyUnicode_FromString(doc_str)));
  } else if (PyInstanceMethod_Check(obj)) {
    auto* f = (PyCFunctionObject*)(PyInstanceMethod_Function(obj));
    f->m_ml->ml_doc = doc_str;
  } else {
    THROW(RuntimeError) << "function is " << Py_TYPE(obj)->tp_name << ", not a valid function";
  }
  f.inc_ref();
  return f;
}

py::object ReplaceDoc(py::object f, const std::string& doc_string) {
  static std::vector<std::string> all_doc_strings;
  all_doc_strings.emplace_back(doc_string);
  const char* doc_str = all_doc_strings.back().c_str();
  PyObject* obj = f.ptr();
  if (PyCFunction_Check(obj)) {
    auto* f = (PyCFunctionObject*)obj;
    if (!f->m_ml->ml_doc) {
      THROW(RuntimeError) << "function " << f->m_ml->ml_name << " has not a docstring yet.";
    }
    f->m_ml->ml_doc = doc_str;
  } else if (PyFunction_Check(obj)) {
    auto* f = (PyFunctionObject*)obj;
    if (f->func_doc == Py_None) {
      THROW(RuntimeError) << "function "
                          << PyBytes_AsString(
                                 PyUnicode_AsEncodedString(f->func_name, "utf-8", "~E~"))
                          << " has not a docstring yet.";
    }
    Py_DECREF(f->func_doc);
    f->func_doc = PyUnicode_FromString(doc_str);
  } else {
    THROW(RuntimeError) << "function is " << Py_TYPE(obj)->tp_name << ", not a valid function.";
  }
  f.inc_ref();
  return f;
}

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("add_doc", &oneflow::AddFunctionDoc);
  m.def("reset_doc", &oneflow::ReplaceDoc);
}
