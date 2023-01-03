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

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#if PY_VERSION_HEX < 0x03070000
#define PYGETSET_NAME(name) const_cast<char*>(name)
#else
#define PYGETSET_NAME(name) (name)
#endif

using functional::PyObjectPtr;

#define INFO_FLOAT_TYPE_SEQ FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ BFLOAT16_DATA_TYPE_SEQ
#define INFO_TYPE_SEQ INT_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ INFO_FLOAT_TYPE_SEQ

template<typename>
struct is_floating_point_with_half : public std::false_type {};

#define DEFINE_IS_FLOATING_POINT_WITH_HALF(cpp_type, of_datatype) \
  template<>                                                      \
  struct is_floating_point_with_half<cpp_type> : public std::true_type {};

OF_PP_FOR_EACH_TUPLE(DEFINE_IS_FLOATING_POINT_WITH_HALF, INFO_FLOAT_TYPE_SEQ);
#undef DEFINE_IS_FLOATING_POINT_WITH_HALF

template<typename T>
typename std::enable_if<is_floating_point_with_half<T>::value, PyObject*>::type PyGetVal(T value) {
  return PyFloat_FromDouble(value);
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, PyObject*>::type PyGetVal(T value) {
  return PyLong_FromLong(value);
}

PyObject* PyGetMaxVal(DataType datatype) {
#define GET_MAX_VAL(cpp_type, of_datatype) \
  case of_datatype: return PyGetVal(std::numeric_limits<DataTypeToType<of_datatype>>::max());

  switch (datatype) {
    OF_PP_FOR_EACH_TUPLE(GET_MAX_VAL, INFO_TYPE_SEQ);
    default: return NULL;
#undef GET_MAX_VAL
  }
}

PyObject* PyGetMinVal(DataType datatype) {
#define GET_MIN_VAL(cpp_type, of_datatype) \
  case of_datatype: return PyGetVal(std::numeric_limits<DataTypeToType<of_datatype>>::lowest());

  switch (datatype) {
    OF_PP_FOR_EACH_TUPLE(GET_MIN_VAL, INFO_TYPE_SEQ);
    default: return NULL;

#undef GET_MIN_VAL
  }
}

#define GET_FLOAT_RESOLUTION(cpp_type, of_datatype) \
  case of_datatype:                                 \
    return PyFloat_FromDouble(                      \
        std::pow(10, -std::numeric_limits<DataTypeToType<of_datatype>>::digits10));

#define GET_FLOAT_EPS(cpp_type, of_datatype) \
  case of_datatype:                          \
    return PyFloat_FromDouble(std::numeric_limits<DataTypeToType<of_datatype>>::epsilon());

#define GET_FLOAT_TINY(cpp_type, of_datatype) \
  case of_datatype:                           \
    return PyFloat_FromDouble(std::numeric_limits<DataTypeToType<of_datatype>>::min());

PyTypeObject PyIInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.iinfo",  // tp_name
    sizeof(PyDTypeInfo),                             // tp_basicsize
};

PyTypeObject PyFInfoType = {
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
  PyObject* dtype_obj = functional::CastToPyObject(DType::Float());
  static const char* keywords[2] = {"type", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:finfo", const_cast<char**>(keywords),
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

static PyObject* PyDInfo_min(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  PyObject* result = PyGetMinVal(datatype);
  if (!result) {
    THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name() << " not supported by "
                        << self->ob_type->tp_name;
  }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyDInfo_max(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  PyObject* result = PyGetMaxVal(datatype);
  if (!result) {
    THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name() << " not supported by "
                        << self->ob_type->tp_name;
  }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_resolution(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
    OF_PP_FOR_EACH_TUPLE(GET_FLOAT_RESOLUTION, INFO_FLOAT_TYPE_SEQ);
    default:
      THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                          << " not supported by oneflow.finfo";
      return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_eps(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
    OF_PP_FOR_EACH_TUPLE(GET_FLOAT_EPS, INFO_FLOAT_TYPE_SEQ);
    default:
      THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                          << " not supported by oneflow.finfo";
      return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_tiny(PyObject* self, void*) {
  HANDLE_ERRORS
  DataType datatype = PyDTypeInfo_UnpackDataType(self);
  switch (datatype) {
    OF_PP_FOR_EACH_TUPLE(GET_FLOAT_TINY, INFO_FLOAT_TYPE_SEQ);
    default:
      THROW(RuntimeError) << PyDTypeInfo_UnpackDType(self)->name()
                          << " not supported by oneflow.finfo";
      return NULL;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyDInfo_dtype(PyObject* self, void*) {
  HANDLE_ERRORS
  std::string name = ((PyDTypeInfo*)self)->dtype->name();
  name = name.erase(0, name.find('.') + 1);
  return PyUnicode_FromString(name.data());
  END_HANDLE_ERRORS
}

static PyObject* PyIInfo_str(PyObject* self) {
  HANDLE_ERRORS
  std::ostringstream oss;
  oss << "iinfo(min=" << PyLong_AS_LONG(PyDInfo_min((PyObject*)self, NULL)) << ", ";
  oss << "max=" << PyLong_AS_LONG(PyDInfo_max((PyObject*)self, NULL)) << ", ";
  oss << "dtype=" << PyDTypeInfo_UnpackDType(self)->name() << ", ";
  oss << "bits=" << PyLong_AS_LONG(PyDInfo_bits((PyObject*)self, NULL)) << ")";
  return PyUnicode_FromString(oss.str().data());
  END_HANDLE_ERRORS
}

static PyObject* PyFInfo_str(PyObject* self) {
  HANDLE_ERRORS
  std::ostringstream oss;
  oss << "finfo(resolution=" << PyFloat_AS_DOUBLE(PyFInfo_resolution((PyObject*)self, NULL))
      << ", ";
  oss << "min=" << PyFloat_AS_DOUBLE(PyDInfo_min((PyObject*)self, NULL)) << ", ";
  oss << "max=" << PyFloat_AS_DOUBLE(PyDInfo_max((PyObject*)self, NULL)) << ", ";
  oss << "eps=" << PyFloat_AS_DOUBLE(PyFInfo_eps((PyObject*)self, NULL)) << ", ";
  oss << "tiny=" << PyFloat_AS_DOUBLE(PyFInfo_tiny((PyObject*)self, NULL)) << ", ";
  oss << "dtype=" << PyDTypeInfo_UnpackDType(self)->name() << ", ";
  oss << "bits=" << PyLong_AS_LONG(PyDInfo_bits((PyObject*)self, NULL)) << ")";
  return PyUnicode_FromString(oss.str().data());
  END_HANDLE_ERRORS
}

static struct PyGetSetDef PyIInfo_properties[] = {
    {PYGETSET_NAME("bits"), (getter)PyDInfo_bits, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("max"), (getter)PyDInfo_max, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("min"), (getter)PyDInfo_min, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("dtype"), (getter)PyDInfo_dtype, nullptr, nullptr, nullptr},
    {nullptr},
};

static struct PyGetSetDef PyFInfo_properties[] = {
    {PYGETSET_NAME("bits"), (getter)PyDInfo_bits, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("max"), (getter)PyDInfo_max, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("min"), (getter)PyDInfo_min, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("resolution"), (getter)PyFInfo_resolution, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("eps"), (getter)PyFInfo_eps, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("tiny"), (getter)PyFInfo_tiny, nullptr, nullptr, nullptr},
    {PYGETSET_NAME("dtype"), (getter)PyDInfo_dtype, nullptr, nullptr, nullptr},
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
#undef GET_FLOAT_RESOLUTION
#undef GET_FLOAT_EPS
#undef GET_FLOAT_TINY
#undef INFO_FLOAT_TYPE_SEQ
#undef INFO_TYPE_SEQ
#undef PYGETSET_NAME