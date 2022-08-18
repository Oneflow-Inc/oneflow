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

#include <limits>
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/framework/typeinfo.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()
using functional::PyObjectPtr;

#define GET_INT_MAX_VAL(datatype) \
  case DataType::datatype:        \
    return PyLong_FromLong(GetMaxVal<DataTypeToType<DataType::datatype>>());
#define GET_INT_MIN_VAL(datatype) \
  case DataType::datatype:        \
    return PyLong_FromLong(GetMinVal<DataTypeToType<DataType::datatype>>());
#define GET_FLOAT_MAX_VAL(datatype) \
  case DataType::datatype:          \
    return PyFloat_FromDouble(GetMaxVal<DataTypeToType<DataType::datatype>>());
#define GET_FLOAT_MIN_VAL(datatype) \
  case DataType::datatype:          \
    return PyFloat_FromDouble(GetMinVal<DataTypeToType<DataType::datatype>>());
#define GET_FLOAT_RESOLUTION(datatype) \
  case DataType::datatype: \
    return PyFloat_FromDouble( \
        std::pow(10, -std::numeric_limits<DataTypeToType<DataType::datatype>>::digits10));

#define INT_TYPE (kUInt8)(kInt8)(kInt32)(kInt64)
#define FLOAT_TYPE (kFloat)(kDouble)

static PyTypeObject PyIInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.iinfo",  // tp_name
    sizeof(PyDTypeInfo),                             // tp_basicsize
};

static PyTypeObject PyFInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.finfo",  // tp_name
    sizeof(PyDTypeInfo),                             // tp_basicsize
};

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
  CHECK_OR_THROW(!self->dtype->is_floating_point() && !self->dtype->is_complex())
      << Error::TypeError()
      << "oneflow.iinfo() requires an integer input type. Use oneflow.finfo to handle '"
      << self->dtype->name() << "' ";
  return (PyObject*)self;
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_new(PyTypeObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* dtype_obj = NULL;
  static const char* keywords[2] = {"type", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:finfo", const_cast<char**>(keywords),
                                   &dtype_obj)) {
    return NULL;
  }
  CHECK_OR_THROW(functional::PyDTypeCheck(dtype_obj))
      << Error::TypeError() << "finfo(): argument 'type' must be oneflow.dtype, but found "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(dtype_obj)));

  auto* self = (PyDTypeInfo*)PyFInfoType.tp_alloc(&PyFInfoType, 0);
  if (!self) { throw py::error_already_set(); }
  self->dtype = functional::PyUnpackDType(dtype_obj);
  CHECK_OR_THROW(self->dtype->is_floating_point() && !self->dtype->is_complex())
      << Error::TypeError()
      << "oneflow.finfo() requires a float input type. Use oneflow.iinfo to handle '"
      << self->dtype->name() << "' ";
  return (PyObject*)self;
  END_HANDLE_ERRORS
}

static PyObject* PyDInfo_bits(PyObject* self, void*) {
  HANDLE_ERRORS
  size_t bits = ASSERT(((PyDTypeInfo*)self)->dtype->bytes()) * 8;
  return PyLong_FromSize_t(bits);
  END_HANDLE_ERRORS
}

static PyObject* PyIInfo_min(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
    OF_PP_FOR_EACH_ATOMIC(GET_INT_MIN_VAL, INT_TYPE);
    default:
      THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                          << " not supported by oneflow.iinfo";
      return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_min(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
    OF_PP_FOR_EACH_ATOMIC(GET_FLOAT_MIN_VAL, FLOAT_TYPE);
    default:
      THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                          << " not supported by oneflow.finfo";
      return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyIInfo_max(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
    OF_PP_FOR_EACH_ATOMIC(GET_INT_MAX_VAL, INT_TYPE);
    default:
      THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                          << " not supported by oneflow.iinfo";
      return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_max(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
    OF_PP_FOR_EACH_ATOMIC(GET_FLOAT_MAX_VAL, FLOAT_TYPE);
    default:
      THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                          << " not supported by oneflow.finfo";
      return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_resolution(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
  OF_PP_FOR_EACH_ATOMIC(GET_FLOAT_RESOLUTION, FLOAT_TYPE);
  default:
    THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                        << " not supported by oneflow.finfo";
    return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyDInfo_dtype(PyObject* self, void*) {
  HANDLE_ERRORS
  Symbol<DType> dtype = ((PyDTypeInfo*)self)->dtype;
  return functional::CastToPyObject(dtype);
  END_HANDLE_ERRORS
}

static PyObject* PyIInfo_str(PyObject* self) {
  HANDLE_ERRORS
  std::ostringstream oss;
  oss << "iinfo(min=" << PyLong_AS_LONG(PyIInfo_min((PyObject*)self, NULL)) << ", ";
  oss << "max=" << PyLong_AS_LONG(PyIInfo_max((PyObject*)self, NULL)) << ", ";
  oss << "dtype=" << PyDTypeInfo_UnpackDType(self)->name() << ", ";
  oss << "bits=" << PyLong_AS_LONG(PyDInfo_bits((PyObject*)self, NULL)) << ")";
  return PyUnicode_FromString(oss.str().data());
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_str(PyObject* self) {
  HANDLE_ERRORS
  std::ostringstream oss;
  oss << "iinfo(min=" << PyFloat_AS_DOUBLE(PyFInfo_min((PyObject*)self, NULL)) << ", ";
  oss << "max=" << PyFloat_AS_DOUBLE(PyFInfo_max((PyObject*)self, NULL)) << ", ";
  oss << "dtype=" << PyDTypeInfo_UnpackDType(self)->name() << ", ";
  oss << "bits=" << PyLong_AS_LONG(PyDInfo_bits((PyObject*)self, NULL)) << ")";
  return PyUnicode_FromString(oss.str().data());
  END_HANDLE_ERRORS
}

static struct PyGetSetDef PyIInfo_properties[] = {
    {"bits", (getter)PyDInfo_bits, nullptr, nullptr, nullptr},
    {"max", (getter)PyIInfo_max, nullptr, nullptr, nullptr},
    {"min", (getter)PyIInfo_min, nullptr, nullptr, nullptr},
    {"dtype", (getter)PyDInfo_dtype, nullptr, nullptr, nullptr},
    {nullptr},
};

static struct PyGetSetDef PyFInfo_properties[] = {
    {"bits", (getter)PyDInfo_bits, nullptr, nullptr, nullptr},
    {"max", (getter)PyFInfo_max, nullptr, nullptr, nullptr},
    {"min", (getter)PyFInfo_min, nullptr, nullptr, nullptr},
    {"resolution", (getter)PyFInfo_resolution, nullptr, nullptr, nullptr},
    {"dtype", (getter)PyDInfo_dtype, nullptr, nullptr, nullptr},
    {nullptr},
};

static void init_info_type() {
  PyIInfoType.tp_flags = Py_TPFLAGS_DEFAULT;
  PyIInfoType.tp_str = (reprfunc)PyIInfo_str;
  PyIInfoType.tp_repr = (reprfunc)PyIInfo_str;
  PyIInfoType.tp_new = (newfunc)PyIInfo_new;
  PyIInfoType.tp_getset = PyIInfo_properties;
  if (PyType_Ready(&PyIInfoType) < 0) { return; }

  PyFInfoType.tp_flags = Py_TPFLAGS_DEFAULT;
  PyFInfoType.tp_str = (reprfunc)PyFInfo_str;
  PyFInfoType.tp_repr = (reprfunc)PyFInfo_str;
  PyFInfoType.tp_new = (newfunc)PyFInfo_new;
  PyFInfoType.tp_getset = PyFInfo_properties;
  if (PyType_Ready(&PyFInfoType) < 0) { return; }
}

ONEFLOW_API_PYBIND11_MODULE("_C", m) {
  init_info_type();
  if (PyModule_AddObject(m.ptr(), "iinfo", (PyObject*)&PyIInfoType) < 0) return;
  if (PyModule_AddObject(m.ptr(), "finfo", (PyObject*)&PyFInfoType) < 0) return;
}

}  // namespace one
}  // namespace oneflow
#undef ASSERT
#undef ASSERT_PTR
