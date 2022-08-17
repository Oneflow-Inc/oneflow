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

#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/framework/typeinfo.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()
using functional::PyObjectPtr;

static PyTypeObject PyIInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.iinfo",  // tp_name
    sizeof(PyDTypeInfo),                             // tp_basicsize
};

static PyObject* PyIInfo_str(PyIInfo* self) {
  HANDLE_ERRORS
  std::string result = "iinfo(";
  result += "dtype=";
  result += self->dtype->name();
  result += ")";
  return PyUnicode_FromString(result.data());
  END_HANDLE_ERRORS
}

static PyObject* PyIInfo_new(PyTypeObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* dtype_obj = NULL;
  static const char* keywords[2] = {"type", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:iinfo", const_cast<char**>(keywords),
                                   &dtype_obj)) {
    return NULL;
  }
  CHECK_OR_THROW(functional::PyDTypeCheck(dtype_obj))
      << Error::TypeError() << "iinfo(): argument 'type' must be oneflow.dtype, but found "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(dtype_obj)));

  auto* self = (PyDTypeInfo*)PyIInfoType.tp_alloc(&PyIInfoType, 0);
  if (!self) { throw py::error_already_set(); }
  self->dtype = functional::PyUnpackDType(dtype_obj);
  return (PyObject*)self;
  END_HANDLE_ERRORS
}

static PyObject* PyDInfo_bits(PyObject* self, void*) {
  HANDLE_ERRORS
  size_t bits = ASSERT(((PyDTypeInfo*)self)->dtype->bytes()) * 8;
  return PyLong_FromSize_t(bits);
  END_HANDLE_ERRORS
}

static PyObject* PyDInfo_dtype(PyObject* self, void*) {
  HANDLE_ERRORS
  Symbol<DType> dtype = ((PyDTypeInfo*)self)->dtype;
  return functional::CastToPyObject(dtype);
  END_HANDLE_ERRORS
}

static struct PyGetSetDef PyIInfo_properties[] = {
    // TODO(WangYi): add max / min / eps
    {"bits", (getter)PyDInfo_bits, nullptr, nullptr, nullptr},
    {"dtype", (getter)PyDInfo_dtype, nullptr, nullptr, nullptr},
    {nullptr},
};

static void init_iinfo_type() {
  PyIInfoType.tp_flags = Py_TPFLAGS_DEFAULT;
  PyIInfoType.tp_str = (reprfunc)PyIInfo_str;
  PyIInfoType.tp_repr = (reprfunc)PyIInfo_str;
  PyIInfoType.tp_new = (newfunc)PyIInfo_new;
  PyIInfoType.tp_getset = PyIInfo_properties;
  if (PyType_Ready(&PyIInfoType) < 0) { return; }
}

ONEFLOW_API_PYBIND11_MODULE("_C", m) {
  init_iinfo_type();
  if (PyModule_AddObject(m.ptr(), "iinfo", (PyObject*)&PyIInfoType) < 0) return;
}

}  // namespace one
}  // namespace oneflow
#undef ASSERT
#undef ASSERT_PTR
