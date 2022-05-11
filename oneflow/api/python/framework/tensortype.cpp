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
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/api/python/exception/exception.h"

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
    // functional::To failed when dtype->datatype() == kChar
    // {kChar, "CharTensor"},
    {kFloat, "FloatTensor"},
    {kDouble, "DoubleTensor"},
    {kInt8, "CharTensor"},
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

static PyObject* PyTensortypeMetaCls_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto* temp = functional::_legacy_tensor_ctor(NULL, args, kwargs);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  PyTensorObject* tensor = (PyTensorObject*)PyTensorObject_Type->tp_alloc(PyTensorObject_Type, 0);
  tensor->data = PyTensor_Unpack(temp);
  tensor->data->set_pyobject(self);

  PyTensortype* tensortype = (PyTensortype*)self;
  const Optional<std::string>& device =
      ASSERT(DeviceTag4DeviceType(PyTensortype_UnpackDevice((PyObject*)tensortype)));
  const auto& t =
      functional::To(tensor->data, device, PyTensortype_UnpackDType((PyObject*)tensortype), false);
  tensor->data = CHECK_JUST(t);

  // reset temp data to prevent clearing the pyobject
  // when the temp is deallocated
  ((PyTensorObject*)temp)->data.reset();
  Py_XDECREF(temp);
  return (PyObject*)tensor;
  END_HANDLE_ERRORS
};

static std::string datatype_to_string(DataType datatype) {
  CHECK_OR_THROW(datatype_to_string_dict.find(datatype) != datatype_to_string_dict.end())
      << "unsupported datatype";
  return datatype_to_string_dict.at(datatype);
}

static std::string devicetype_to_string(DeviceType dtype) {
  CHECK_OR_THROW(devicetype_to_string_dict.find(dtype) != devicetype_to_string_dict.end())
      << "unsupported devicetype";
  return devicetype_to_string_dict.at(dtype);
}

PyObject* PyTensortype_FromString(const std::string& tensortype_str) {
  std::string oneflow_prefix = "oneflow.";
  auto mismatch_pair =
      std::mismatch(oneflow_prefix.begin(), oneflow_prefix.end(), tensortype_str.begin());
  CHECK_OR_THROW(mismatch_pair.first == oneflow_prefix.end()) << "invalid type: " << tensortype_str;

  std::string dtype_str = tensortype_str.substr(oneflow_prefix.size());
  auto it = std::find_if(
      tensortype_list.begin(), tensortype_list.end(),
      [dtype_str](PyTensortype* type) { return std::string(type->name) == dtype_str; });
  CHECK_OR_THROW(it != tensortype_list.end()) << "invalid type: " << tensortype_str;
  return (PyObject*)(*it);
}

static std::string get_name(DataType datatype, DeviceType device) {
  auto device_string = devicetype_to_string(device);
  if (device_string.empty()) return datatype_to_string(datatype);
  return device_string + "." + datatype_to_string(datatype);
}

static std::string get_doc(PyTensortype* tensortype) {
  std::cout << "datatype in get_doc: " << tensortype->datatype << std::endl;
  std::string dtype_str = ASSERT(DType::Get(tensortype->datatype))->name();
  dtype_str = dtype_str.substr(dtype_str.rfind("."));
  std::string device = tensortype->is_cuda ? "cuda" : "cpu";
  std::ostringstream ss;
  ss << "Creates a Tensor with the dtype of "<< dtype_str << " and the device on "<< device <<" , it has the same parameters as :func:`oneflow.Tensor`";
  return ss.str();
}

static void init_metaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_base = &PyType_Type;
  metaclass.tp_call = PyTensortypeMetaCls_call;
  if (PyType_Ready(&metaclass) < 0) { return; }
}

static void init_tensortype(PyTypeObject& type, PyTypeObject& type_template,
                            const std::string& name, const std::string& doc) {
  memcpy(&type, &type_template, sizeof(PyTypeObject));
  char* tp_name = new char[64]{'\0'};

  // name.c_str() has bug here, so convert with iterating
  for (int i = 0; i < name.size(); i++) tp_name[i] = name[i];
  type.tp_name = tp_name;
  type.tp_doc = doc.c_str();
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  if (PyType_Ready(&type) < 0) {
    CHECK_OR_THROW(false) << "tensortype initialization failed";
    return;
  }
}

static void generalize_tensortype_list() {
  init_metaclass(PyTensortypeMetaClass);
  for (const auto& datatype_string_pair : datatype_to_string_dict) {
    for (const auto& devicetype_string_pair : devicetype_to_string_dict) {
      PyTensortype* tensortype = new PyTensortype();

      // set name
      std::string name = get_name(datatype_string_pair.first, devicetype_string_pair.first);

      size_t n = sizeof(tensortype->name);
      strncpy(tensortype->name, name.c_str(), n);
      tensortype->name[n - 1] = '\0';

      // set type
      tensortype->datatype = datatype_string_pair.first;
      tensortype->device = devicetype_string_pair.first;
      tensortype->is_cuda = tensortype->device == DeviceType::kCUDA;
      tensortype_list.push_back(tensortype);

      name = "oneflow." + name;
      std::cout << "datatype: " << tensortype->datatype << std::endl;
      std::string doc = get_doc(tensortype);
      init_tensortype(tensortype->py_type, PyTensortypeTemplate, name, doc);
    }
  }
}

static void binding(pybind11::module_& m) {
  generalize_tensortype_list();
  for (PyTensortype* tensortype : tensortype_list) {
    Py_INCREF(tensortype);
    auto module = m.def_submodule("_C");
    if (tensortype->is_cuda) module = module.def_submodule("cuda");

    std::string type_name = std::string(tensortype->name);
    type_name = type_name.substr(type_name.rfind('.') + 1);
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

PyObject* PyTensortype_FromDTypeDeviceType(DataType datatype, DeviceType device) {
  auto it = std::find_if(tensortype_list.begin(), tensortype_list.end(),
                         [datatype, device](PyTensortype* x) {
                           return (x->datatype == datatype) && (x->device == device);
                         });
  if (it == tensortype_list.end()) return PyErr_Format(PyExc_RuntimeError, "unsupported dtype");
  return (PyObject*)(*it);
};

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) { binding(m); }