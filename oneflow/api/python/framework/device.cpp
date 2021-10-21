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
#include <pybind11/operators.h>
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/api/python/framework/device.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace py = pybind11;

namespace oneflow {

/* static */ void DeviceExportUtil::CheckDeviceType(const std::string& type) {
  if (Device::type_supported.find(type) == Device::type_supported.end()) {
    std::string error_msg =
        "Expected one of cpu, cuda device type at start of device string " + type;
    throw std::runtime_error(error_msg);
  }
}

/* static */ Symbol<Device> DeviceExportUtil::ParseAndNew(const std::string& type_and_id) {
  std::string type;
  int device_id = -1;
  ParsingDeviceTag(type_and_id, &type, &device_id).GetOrThrow();
  if (device_id == -1) {
    return New(type);
  } else {
    return New(type, device_id);
  }
}

/* static */ Symbol<Device> DeviceExportUtil::New(const std::string& type) {
  CheckDeviceType(type);
  return Device::New(type).GetOrThrow();
}

/* static */ Symbol<Device> DeviceExportUtil::New(const std::string& type, int64_t device_id) {
  CheckDeviceType(type);
  return Device::New(type, device_id).GetOrThrow();
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<Device>, std::shared_ptr<Symbol<Device>>>(m, "device")
      .def(py::init([](const std::string& type_and_id) {
        return DeviceExportUtil::ParseAndNew(type_and_id);
      }))
      .def(py::init([](const std::string& type, int64_t device_id) {
        return DeviceExportUtil::New(type, device_id);
      }))
      .def_property_readonly("type", [](const Symbol<Device>& d) { return d->type(); })
      .def_property_readonly("index", [](const Symbol<Device>& d) { return d->device_id(); })
      .def("__str__", [](const Symbol<Device>& d) { return d->ToString(); })
      .def("__repr__", [](const Symbol<Device>& d) { return d->ToRepr(); })
      .def(py::self == py::self)
      .def(py::hash(py::self));
}

}  // namespace oneflow
