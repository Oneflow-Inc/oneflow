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
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"
#include "oneflow/core/framework/tmp_compute_stream_type_guard.h"

ONEFLOW_API_PYBIND11_MODULE("eager", m) {
  using namespace oneflow;
  namespace py = pybind11;
  m.def(
      "Sync", []() { return vm::ClusterSync(); }, py::call_guard<py::gil_scoped_release>());

  py::class_<one::DevVmDepObjectConsumeModeGuard,
             std::shared_ptr<one::DevVmDepObjectConsumeModeGuard>>(
      m, "DevVmDepObjectConsumeModeGuard");

  m.def("SourceOpOnlyResourceDependenceModeGuard", []() {
    return std::make_shared<one::DevVmDepObjectConsumeModeGuard>(
        one::DevVmDepObjectConsumeMode::NONE);
  });

  py::class_<TmpComputeStreamTypeGuard, std::shared_ptr<TmpComputeStreamTypeGuard>>(
      m, "TmpComputeStreamTypeGuard")
      .def(py::init([](const Optional<std::string>& stream_tag) {
        if (stream_tag.has_value()) {
          return std::make_shared<TmpComputeStreamTypeGuard>(*CHECK_JUST(stream_tag));
        } else {
          return std::make_shared<TmpComputeStreamTypeGuard>();
        }
      }));
}
