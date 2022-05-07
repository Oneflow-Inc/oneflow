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
#include <methodobject.h>
#include <object.h>
#include <objimpl.h>
#include <pybind11/pybind11.h>
#include <strings.h>
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
  char name[32];
  bool is_cuda;
  DataType dtype;
  DeviceType device;
} PyTensortype;

std::vector<DataType> DataType_list {
  kChar, kFloat, kDouble, kInt8, kInt32, kInt64, kUInt8, kFloat16, kBFloat16, kBool 
};

std::vector<DeviceType> DeviceType_list {
  DeviceType::kCPU, DeviceType::kCUDA,
};


static PyTypeObject PyTensortypeMetaClass {
  PyVarObject_HEAD_INIT(NULL, 0)
  "oneflow.tensortype", // tp_name
  sizeof(PyTypeObject), // tp_basicsize
};

static PyTypeObject PyTensortypeTemplate {
  PyVarObject_HEAD_INIT(&PyTensortypeMetaClass, 0)
  NULL, // tp_name
  sizeof(PyTensortype), // tp_basicsize
};

std::unordered_map<DataType, std::string> dtype_to_string_dict {
  {kChar, "CharTensor"}, {kFloat, "FloatTensor"}, {kDouble, "DoubleTensor"}, {kInt8, "CharTensor"}, {kInt32, "IntTensor"}, {kInt64, "LongTensor"}, {kUInt8, "ByteTensor"},
  {kFloat16, "HalfTensor"}, {kBFloat16, "BFloat16Tensor"}, {kBool, "BoolTensor"}
};

std::unordered_map<DeviceType, std::string> device_to_string_dict {
  {kCPU, "cpu"}, {kCUDA, "cuda"},
};

std::string dtype_to_string(DataType dtype) {
  CHECK_OR_THROW(dtype_to_string_dict.find(dtype) != dtype_to_string_dict.end());
  return dtype_to_string_dict.at(dtype);
}

std::string device_to_string(DeviceType dtype) {
  CHECK_OR_THROW(device_to_string_dict.find(dtype) != device_to_string_dict.end());
  return device_to_string_dict.at(dtype);
}

static std::string get_name(DataType dtype, DeviceType device) {
  return dtype_to_string(dtype) + "_" + device_to_string(device);
}

void init_metaclass(PyTypeObject& metaclass) {
    metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    metaclass.tp_base = &PyType_Type;
    if(PyType_Ready(&metaclass) < 0) 
    {
        std::cout << "metaclass init failed" << std::endl;
        return;
    }
    std::cout << "metaclass init successfully" << std::endl;
}

void init_tensortype(PyTypeObject& type, PyTypeObject& type_template, const char* name) {
  memcpy(&type, &type_template, sizeof(PyTypeObject));
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_new = PyType_Type.tp_new;
  if(PyType_Ready(&type) < 0) {
      std::cout << "error in init tensortype" << std::endl;
  }
}

std::vector<PyTensortype*> tensortype_list;

void generalize_tensortype_list()
{
    init_metaclass(PyTensortypeMetaClass);
    for(DataType dtype: DataType_list)
    {
        for(DeviceType device: DeviceType_list)
        {
            std::cout << "dealing with " << dtype << " and " << device << std::endl;
            PyTensortype* tensortype = new PyTensortype();
            const char* s = "test";
            init_tensortype(tensortype->py_type, PyTensortypeTemplate, s);

            // set name
            strncpy(tensortype->name, get_name(dtype, device).c_str(), sizeof(tensortype->name));
            tensortype->name[sizeof(tensortype->name) - 1] = '\0';

            // set type
            tensortype->dtype = dtype;
            tensortype->device = device;
            tensortype->is_cuda = true;
            tensortype_list.push_back(tensortype);
        }
    }
}

static void binding(pybind11::module_& m) {
  generalize_tensortype_list();
  for (PyTensortype* tensortype: tensortype_list) {
    PyObject_SetAttrString((PyObject*)tensortype, "__module__", PyUnicode_FromString("oneflow"));
    if (tensortype && PyModule_AddObject(m.ptr(), tensortype->name, (PyObject*)tensortype) < 0) {
      std::cout << "failed" << std::endl;
       CHECK_OR_THROW(false); 
    }
    std::cout << tensortype->name << std::endl;
  }
  std::cout << "all finished" << std::endl;
}


}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  binding(m);
}