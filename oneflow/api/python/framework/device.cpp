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

namespace py = pybind11;

namespace oneflow {

namespace {

struct DeviceExportUtil final {
  static bool IsTypeSupported(const std::string& type) {
    return DeviceExportUtil::type_supported.find(type) != DeviceExportUtil::type_supported.end();
  }

  static bool IsValidDeviceId(const std::string& device_id) {
    if (IsStrInt(device_id)) { return device_id.at(0) != '-'; }
    return false;
  }

  static int CheckAndGetDeviceId(const std::string& type_and_id, const std::string& type,
                                 std::string::size_type pos) {
    int device_id = 0;
    if (pos < type_and_id.size()) {
      std::string id = type_and_id.substr(pos + 1);
      if (!IsValidDeviceId(id)) {
        throw std::runtime_error("Invalid device string: " + type_and_id);
      }
      device_id = std::stoi(id);
      if (type == "cpu" && device_id > 0) {
        throw std::runtime_error("CPU device index must be 0");
      }
    }
    return device_id;
  }

  static std::shared_ptr<Device> MakeDevice(const std::string& type_and_id) {
    std::string::size_type pos = type_and_id.find(':');
    if (pos == std::string::npos) { pos = type_and_id.size(); }
    std::string type = type_and_id.substr(0, pos);
    if (!IsTypeSupported(type)) {
      std::string error_msg =
          "Expected one of cpu, cuda device type at start of device string " + type;
      throw std::runtime_error(error_msg);
    }
    int device_id = CheckAndGetDeviceId(type_and_id, type, pos);
    return std::make_shared<Device>(type, device_id);
  }

  static const std::unordered_set<std::string> type_supported;
};

const std::unordered_set<std::string> DeviceExportUtil::type_supported({"cuda", "cpu"});

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Device, std::shared_ptr<Device>>(m, "device")
      .def(py::init(&DeviceExportUtil::MakeDevice))
      .def_property_readonly("type", &Device::type)
      .def_property_readonly("index", &Device::device_id)
      .def("__str__", &Device::ToString)
      .def("__repr__", &Device::ToString);
}

}  // namespace oneflow
