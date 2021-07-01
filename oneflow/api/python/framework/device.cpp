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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace py = pybind11;

namespace oneflow {

namespace {

struct DeviceExportUtil final {
  static Symbol<Device> MakeDevice(const std::string& type_and_id) {
    std::string::size_type pos = type_and_id.find(':');
    if (pos == std::string::npos) { pos = type_and_id.size(); }
    std::string type = type_and_id.substr(0, pos);
    int device_id = type == "cpu" ? 0 : GlobalProcessCtx::LocalRank();
    if (pos < type_and_id.size()) {
      std::string id = type_and_id.substr(pos + 1);
      if (!IsStrInt(id)) { throw std::runtime_error("Invalid device string: " + type_and_id); }
      device_id = std::stoi(id);
      if (type == "cpu" && device_id != 0) {
        throw std::runtime_error("CPU device index must be 0");
      }
    }
    return MakeDevice(type, device_id);
  }

  static Symbol<Device> MakeDevice(const std::string& type, int64_t device_id) {
    if (Device::type_supported.find(type) == Device::type_supported.end()) {
      std::string error_msg =
          "Expected one of cpu, cuda device type at start of device string " + type;
      throw std::runtime_error(error_msg);
    }
    return Device::New(type, device_id).GetOrThrow();
  }
};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<Device>, std::shared_ptr<Symbol<Device>>>(m, "device")
      .def(py::init(
          [](const std::string& type_and_id) { return DeviceExportUtil::MakeDevice(type_and_id); }))
      .def(py::init([](const std::string& type, int64_t device_id) {
        return DeviceExportUtil::MakeDevice(type, device_id);
      }))
      .def_property_readonly("type", [](const Symbol<Device>& d) { return d->type(); })
      .def_property_readonly("index", [](const Symbol<Device>& d) { return d->device_id(); })
      .def("__eq__", [](const Symbol<Device>& d1, const Symbol<Device>& d2) { return *d1 == *d2; })
      .def("__str__", [](const Symbol<Device>& d) { return d->ToString(); })
      .def("__repr__", [](const Symbol<Device>& d) { return d->ToRepr(); });
}

}  // namespace oneflow
