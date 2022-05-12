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
#include <pybind11/pybind11.h>
#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/api/python/framework/tensortype.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/api/python/exception/exception.h"

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

static PyTypeObject PyTensorTypeMetaClass{
    PyVarObject_HEAD_INIT(NULL, 0) "oneflow.tensortype",  // tp_name
    sizeof(PyTypeObject),                                 // tp_basicsize
};

static PyTypeObject PyTensorTypeTemplate{
    PyVarObject_HEAD_INIT(&PyTensorTypeMetaClass, 0) NULL,  // tp_name
    sizeof(PyTensorType),                                   // tp_basicsize
};

static std::vector<PyTensorType*> tensor_types;

static std::vector<std::pair<DataType, std::string>> all_data_types = {
    {kFloat, "FloatTensor"},  {kDouble, "DoubleTensor"},     {kInt8, "CharTensor"},
    {kInt32, "IntTensor"},    {kInt64, "LongTensor"},        {kUInt8, "ByteTensor"},
    {kFloat16, "HalfTensor"}, {kBFloat16, "BFloat16Tensor"}, {kBool, "BoolTensor"},
};

static std::vector<std::pair<DeviceType, std::string>> all_device_types = {
    {kCPU, "oneflow"},
    {kCUDA, "oneflow.cuda"},
};

static PyObject* PyTensorTypeMetaCls_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto* tensor = functional::_legacy_tensor_ctor(NULL, args, kwargs);
  if (PyErr_Occurred()) { throw py::error_already_set(); }

  Optional<std::string> device = ASSERT(DeviceTag4DeviceType(PyTensorType_UnpackDevice(self)));
  const auto& data_type = PyTensorType_UnpackDType(self);
  return PyTensor_New(
      ASSERT_PTR(functional::To(PyTensor_Unpack(tensor), device, data_type, /*copy=*/false)));
  END_HANDLE_ERRORS
};

PyObject* PyTensorType_FromString(const std::string& tensortype) {
  auto it = std::find_if(
      tensor_types.begin(), tensor_types.end(),
      [tensortype](PyTensorType* type) { return std::string(type->name) == tensortype; });
  if (it == tensor_types.end()) {
    return PyErr_Format(PyExc_ValueError, "invalid type: %s", tensortype.data());
  }
  return (PyObject*)(*it);
}

static const char* get_doc(PyTensorType* tensortype) {
  // all tensortype docs
  static std::vector<std::string> tensortype_doc;

  std::string dtype /* = PyTensorType_UnpackDType(tensortype)->name()*/;
  std::string device = ASSERT(DeviceTag4DeviceType(tensortype->device));
  std::string doc = "Creates a Tensor with the dtype of " + dtype + " and the device on " + device
                    + ", it has the same parameters as :func:`oneflow.Tensor`";
  tensortype_doc.emplace_back(doc);
  return tensortype_doc.back().data();
}

static void init_tensortype_metaclass(PyTypeObject* metaclass) {
  metaclass->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass->tp_base = &PyType_Type;
  metaclass->tp_call = PyTensorTypeMetaCls_call;
  if (PyType_Ready(metaclass) < 0) { return; }
}

static void init_tensortype(PyTypeObject* type, PyTypeObject& type_template, const char* name,
                            const char* doc) {
  memcpy(type, &type_template, sizeof(PyTypeObject));
  type->tp_name = name;
  type->tp_doc = doc;
  type->tp_flags = Py_TPFLAGS_DEFAULT;
  if (PyType_Ready(type) < 0) { THROW(RuntimeError) << "tensortype initialization failed"; }
}

static void generalize_tensor_types() {
  init_tensortype_metaclass(&PyTensorTypeMetaClass);

  for (const auto& devicetype : all_device_types) {
    for (const auto& datatype : all_data_types) {
      PyTensorType* tensortype = new PyTensorType();
      // set name
      std::string name = devicetype.second + "." + datatype.second;
      size_t n = sizeof(tensortype->name);
      strncpy(tensortype->name, name.c_str(), n);
      tensortype->name[n - 1] = '\0';

      // set type
      tensortype->datatype = datatype.first;
      tensortype->device = devicetype.first;
      tensortype->is_cuda = tensortype->device == DeviceType::kCUDA;
      tensor_types.push_back(tensortype);

      const char* doc = get_doc(tensortype);
      init_tensortype(&tensortype->py_type, PyTensorTypeTemplate, tensortype->name, doc);
    }
  }
}

bool PyTensorType_Check(PyObject* obj) { return PyObject_TypeCheck(obj, &PyTensorTypeMetaClass); }

PyObject* PyTensorType_FromDTypeAndDeviceType(DataType datatype, DeviceType device) {
  auto it =
      std::find_if(tensor_types.begin(), tensor_types.end(), [datatype, device](PyTensorType* x) {
        return (x->datatype == datatype) && (x->device == device);
      });
  if (it == tensor_types.end())
    return PyErr_Format(PyExc_ValueError, "unsupported data type (%s) or device (%s)",
                        ASSERT(DType::Get(datatype))->name(), ASSERT(DeviceTag4DeviceType(device)));
  return (PyObject*)(*it);
};

}  // namespace one
}  // namespace oneflow

#undef ASSERT

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("_C", m) {
  static std::string oneflow_prefix = "oneflow.";
  generalize_tensor_types();

  for (PyTensorType* tensortype : tensor_types) {
    Py_INCREF(tensortype);
    std::string name = std::string(tensortype->name);
    size_t idx = name.rfind('.');
    std::string type_name = name.substr(idx + 1);

    name = name.substr(0, idx - 1);
    std::string module_name = name.substr(oneflow_prefix.size());
    auto module = m;
    if (!module_name.empty()) { module = m.def_submodule(module_name.data()); }
    if (tensortype
        && PyModule_AddObject(module.ptr(), type_name.c_str(), (PyObject*)tensortype) < 0) {
      return;
    }
  }
}
