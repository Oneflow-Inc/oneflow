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
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/parallel_conf_util.h"

namespace oneflow {

namespace {

std::tuple<std::string, std::vector<std::string>, std::shared_ptr<cfg::ShapeProto>>
PyGetDeviceTagAndMachineDeviceIdsAndHierarchy(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  return *(GetDeviceTagAndMachineDeviceIdsAndHierarchy(parallel_conf).GetPtrOrThrow());
}

std::shared_ptr<cfg::ParallelConf> PyMakeParallelConf(
    const std::string& device_tag, const std::vector<std::string>& machine_device_ids,
    const std::shared_ptr<Shape>& hierarchy) {
  return MakeParallelConf(device_tag, machine_device_ids, hierarchy).GetPtrOrThrow();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("GetDeviceTagAndMachineDeviceIdsAndHierarchy",
        &PyGetDeviceTagAndMachineDeviceIdsAndHierarchy);
  m.def("MakeParallelConf", &PyMakeParallelConf);
}

}  // namespace oneflow
