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
#include <cstring>
#include <functional>
#include <unordered_map>
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

typedef struct {
  PyTypeObject py_type;
  char name[64];
  bool is_cuda;
  DataType dtype;
  DeviceType device;
} PyTensortype;

// std::vector<DataType> DataType_list{kChar,  kFloat, kDouble,  kInt8,     kInt32,
                                    // kInt64, kUInt8, kFloat16, kBFloat16, kBool};

// std::vector<DeviceType> DeviceType_list{
    // DeviceType::kCPU,
    // DeviceType::kCUDA,
// };

static PyTypeObject PyTensortypeMetaClass{
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.tensortype",  // tp_name
    sizeof(PyTypeObject),                                 // tp_basicsize
};

static PyTypeObject PyTensortypeTemplate{
    PyVarObject_HEAD_INIT(&PyTensortypeMetaClass, 0) NULL,  // tp_name
    sizeof(PyTensortype),                                   // tp_basicsize
};

std::unordered_map<DataType, std::string> dtype_to_string_dict{
    {kChar, "CharTensor"},
    {kFloat, "FloatTensor"},  {kDouble, "DoubleTensor"},
    {kInt8, "Int8Tensor"},  {kInt32, "IntTensor"},    {kInt64, "LongTensor"},
    {kUInt8, "ByteTensor"}, {kFloat16, "HalfTensor"}, {kBFloat16, "BFloat16Tensor"},
    {kBool, "BoolTensor"}
};

std::unordered_map<DeviceType, std::string> device_to_string_dict{
    {kCPU, ""},
    {kCUDA, "cuda"},
};

Maybe<const Symbol<DType>&> TensortypeToDType(PyTensortype* self) {
  return DType::Get(self->dtype);
}

DeviceType TensortypeToDevice(PyTensortype* self) {
  return self->device;
}

static PyObject* TensortypeType_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* tensor = NULL;
  static const char* keywords[2] = {"tensor", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(keywords), &tensor)) {
    return NULL;
  }
  if (!PyTensor_Check(tensor)) { return NULL; }

  Symbol<oneflow::DType> dtype = TensortypeToDType(self);
  Optional<std::string> device_str = ((PyTensortype*)self)->is_cuda ? "cuda": "cpu";

  const auto& t = PyTensor_Unpack(tensor);
  const auto& cast_t = functional::To(t, device_str, dtype, false);
  return functional::CastToPyObject(cast_t);
  END_HANDLE_ERRORS
}

std::string dtype_to_string(DataType dtype) {
  CHECK_OR_THROW(dtype_to_string_dict.find(dtype) != dtype_to_string_dict.end());
  return dtype_to_string_dict.at(dtype);
}

std::string device_to_string(DeviceType dtype) {
  CHECK_OR_THROW(device_to_string_dict.find(dtype) != device_to_string_dict.end());
  return device_to_string_dict.at(dtype);
}

static std::string get_name(DataType dtype, DeviceType device) {
  auto device_string = device_to_string(device);
  if(device_string.empty())
    return dtype_to_string(dtype);
  return  device_string + "." + dtype_to_string(dtype);
}

void init_metaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    return;
  }
}

void init_tensortype(PyTypeObject& type, PyTypeObject& type_template, const std::string& name) {
  memcpy(&type, &type_template, sizeof(PyTypeObject));
  char *tp_name = new char[32]{'\0'};
  // memset(tp_name, '\0', 32);
  for(int i = 0; i < name.size(); i++)
    tp_name[i] = name[i];
  type.tp_name = tp_name;
  type.tp_call = TensortypeType_call;
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_new = PyType_Type.tp_new;
  if (PyType_Ready(&type) < 0) { std::cout << "error in init tensortype" << std::endl; }
}

std::vector<PyTensortype*> tensortype_list;

void generalize_tensortype_list() {
  init_metaclass(PyTensortypeMetaClass);
  for (const auto& datatype_string : dtype_to_string_dict) {
    for (const auto& devicetype_string : device_to_string_dict) {
      PyTensortype* tensortype = new PyTensortype();

      // set name
      std::string name = get_name(datatype_string.first, devicetype_string.first);

      size_t n = sizeof(tensortype->name);
      strncpy(tensortype->name, name.c_str(), n);
      tensortype->name[n - 1] = '\0';

      name = "oneflow." + name;
      init_tensortype(tensortype->py_type, PyTensortypeTemplate, name);

      // set type
      tensortype->dtype = datatype_string.first;
      tensortype->device = devicetype_string.first;
      tensortype->is_cuda = tensortype->device == DeviceType::kCUDA;
      tensortype_list.push_back(tensortype);
    }
  }
}

static void binding(pybind11::module_& m) {
  generalize_tensortype_list();
  for (PyTensortype* tensortype : tensortype_list) {
    tensortype->py_type.tp_setattr = NULL;

    Py_INCREF(tensortype);
    if (tensortype && PyModule_AddObject(m.ptr(), tensortype->name, (PyObject*)tensortype) < 0) {
      CHECK_OR_THROW(false);
    }
  }
}

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) { binding(m); }