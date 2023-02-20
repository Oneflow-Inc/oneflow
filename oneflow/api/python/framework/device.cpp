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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/ep/include/device.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<Device>, std::shared_ptr<Symbol<Device>>>(m, "device")
      .def(py::init([](const std::string& type_or_type_with_device_id) {
        return Device::ParseAndNew(type_or_type_with_device_id).GetOrThrow();
      }))
      .def(py::init([](const std::string& type, int64_t device_id) {
        return Device::New(type, device_id).GetOrThrow();
      }))
      .def(py::init([](const Symbol<Device>& other_device) { return other_device; }))
      .def_property_readonly("type", [](const Symbol<Device>& d) { return d->type(); })
      .def_property_readonly("index", [](const Symbol<Device>& d) { return d->device_id(); })
      .def("__str__", [](const Symbol<Device>& d) { return d->ToString(); })
      .def("__repr__", [](const Symbol<Device>& d) { return d->ToRepr(); })
      .def(py::self == py::self)
      .def(py::hash(py::self));

  m.def(
      "max_alignment_size", []() { return ep::kMaxAlignmentRequirement; },
      py::return_value_policy::copy);
}

}  // namespace oneflow
