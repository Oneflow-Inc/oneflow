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
#include <dictobject.h>
#include <methodobject.h>
#include <object.h>
#include <objimpl.h>
#include <pybind11/pybind11.h>
#include <strings.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>
#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/api/python/framework/tensortype.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.cfg.h"
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

static PyTypeObject PyTensortypeMetaClass{
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.tensortype",  // tp_name
    sizeof(PyTypeObject),                                 // tp_basicsize
};

static PyTypeObject PyTensortypeTemplate{
    PyVarObject_HEAD_INIT(&PyTensortypeMetaClass, 0) NULL,  // tp_name
    sizeof(PyTensortype),                                   // tp_basicsize
};

std::vector<PyTensortype*> tensortype_list;

std::unordered_map<DataType, std::string> datatype_to_string_dict{
    {kChar, "CharTensor"},
    {kFloat, "FloatTensor"},
    {kDouble, "DoubleTensor"},
    {kInt8, "Int8Tensor"},
    {kInt32, "IntTensor"},
    {kInt64, "LongTensor"},
    {kUInt8, "ByteTensor"},
    {kFloat16, "HalfTensor"},
    {kBFloat16, "BFloat16Tensor"},
    {kBool, "BoolTensor"},
    {kComplex32, "ComplexHalfTensor"},
    {kComplex64, "ComplexFloatTensor"},
    {kComplex128, "ComplexDoubleTensor"},
};

std::unordered_map<DeviceType, std::string> devicetype_to_string_dict{
    {kCPU, ""},
    {kCUDA, "cuda"},
};

static PyObject* TensortypeType_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* tensor = NULL;
  static const char* keywords[2] = {"tensor", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(keywords), &tensor)) {
    return NULL;
  }
  if (!PyTensor_Check(tensor)) { return NULL; }

  Symbol<oneflow::DType> dtype = TensortypeToDType(self);
  Optional<std::string> device_str = ((PyTensortype*)self)->is_cuda ? "cuda" : "cpu";
  const auto& t = PyTensor_Unpack(tensor);
  const auto& cast_t = functional::To(t, device_str, dtype, false);
  return functional::CastToPyObject(cast_t);
  END_HANDLE_ERRORS
}

std::string datatype_to_string(DataType datatype) {
  CHECK_OR_THROW(datatype_to_string_dict.find(datatype) != datatype_to_string_dict.end())
      << "unsupported datatype";
  return datatype_to_string_dict.at(datatype);
}

std::string device_to_string(DeviceType dtype) {
  CHECK_OR_THROW(devicetype_to_string_dict.find(dtype) != devicetype_to_string_dict.end())
      << "unsupported devicetype";
  return devicetype_to_string_dict.at(dtype);
}

void tensortype_from_string(const std::string& tensortype_str) {
  std::string prefix("oneflow.");
  CHECK_OR_THROW(std::mismatch(prefix.begin(), prefix.end(), tensortype_str.begin()).first
                 == prefix.end())
      << "invalid type: " << tensortype_str;
}

static std::string get_name(DataType datatype, DeviceType device) {
  auto device_string = device_to_string(device);
  if (device_string.empty()) return datatype_to_string(datatype);
  return device_string + "." + datatype_to_string(datatype);
  // return datatype_to_string(datatype);
}

void init_metaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_base = &PyType_Type;
  metaclass.tp_call = TensortypeType_call;
  if (PyType_Ready(&metaclass) < 0) { return; }
}

void init_tensortype(PyTypeObject& type, PyTypeObject& type_template, const std::string& name) {
  memcpy(&type, &type_template, sizeof(PyTypeObject));
  char* tp_name = new char[64]{'\0'};
  // name.c_str() has bug here, so convert with iterating
  for (int i = 0; i < name.size(); i++) tp_name[i] = name[i];
  type.tp_name = tp_name;
  // type.tp_new = TensortypeType_call;
  type.tp_call = TensortypeType_call;
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  // type.tp_new = PyType_Type.tp_new;
  if (PyType_Ready(&type) < 0) { std::cout << "error in init tensortype" << std::endl; }
}

void generalize_tensortype_list() {
  init_metaclass(PyTensortypeMetaClass);
  for (const auto& datatype_string : datatype_to_string_dict) {
    for (const auto& devicetype_string : devicetype_to_string_dict) {
      PyTensortype* tensortype = new PyTensortype();

      // set name
      std::string name = get_name(datatype_string.first, devicetype_string.first);

      size_t n = sizeof(tensortype->name);
      strncpy(tensortype->name, name.c_str(), n);
      tensortype->name[n - 1] = '\0';

      name = "oneflow." + name;
      init_tensortype(tensortype->py_type, PyTensortypeTemplate, name);

      // set type
      tensortype->datatype = datatype_string.first;
      tensortype->device = devicetype_string.first;
      tensortype->is_cuda = tensortype->device == DeviceType::kCUDA;
      tensortype_list.push_back(tensortype);
    }
  }
}

static void binding(pybind11::module_& m) {
  generalize_tensortype_list();
  for (PyTensortype* tensortype : tensortype_list) {
    Py_INCREF(tensortype);
    auto module = m;
    if (tensortype->is_cuda) module = m.def_submodule("cuda");

    std::string type_name = std::string(tensortype->name);
    type_name = type_name.substr(type_name.rfind('.') + 1);
    // auto module_name = name.substr(0, idx);
    if (tensortype
        && PyModule_AddObject(module.ptr(), type_name.c_str(), (PyObject*)tensortype) < 0) {
      CHECK_OR_THROW(false);
    }
  }
}

bool PyTensortype_Check(PyObject* obj) {
  auto it = std::find_if(tensortype_list.begin(), tensortype_list.end(),
                         [obj](PyTensortype* type) { return obj == (PyObject*)type; });
  return it != tensortype_list.end();
}

PyObject* GetTensortype(DataType datatype, DeviceType device) {
  auto it = std::find_if(tensortype_list.begin(), tensortype_list.end(),
                         [datatype, device](PyTensortype* x) {
                           return (x->datatype == datatype) && (x->device == device);
                         });
  if (it == tensortype_list.end()) return PyErr_Format(PyExc_RuntimeError, "Invalid dtype");
  return (PyObject*)(*it);
};

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) { binding(m); }