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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/to_string.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Device, std::shared_ptr<Device>>(m, "device")
      .def(py::init([](py::str py_device_str, int device_index) {
        std::string device_str = py_device_str.cast<std::string>();
        DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(device_str));
        return std::make_shared<Device>(device_type, device_index);
      }));
}

}  // namespace one
}  // namespace oneflow
