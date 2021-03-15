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
#include "oneflow/core/framework/snapshot_manager.h"

namespace py = pybind11;
namespace oneflow {

namespace {

void InitVariableSnapshotPath(const std::shared_ptr<oneflow::SnapshotManager>& mgr,
                              const std::string& root_dir, bool refresh) {
  mgr->InitVariableSnapshotPath(root_dir, refresh).GetOrThrow();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  using namespace oneflow;
  py::class_<SnapshotManager, std::shared_ptr<SnapshotManager>>(m, "SnapshotManager")
      .def(py::init<>())
      .def("load", &InitVariableSnapshotPath, py::arg("root_dir"), py::arg("refresh") = true)
      .def("get_snapshot_path",
           [](const std::shared_ptr<SnapshotManager>& mgr, const std::string& variable_name) {
             const auto& snapshot_path = mgr->GetSnapshotPath(variable_name).GetOrThrow();
             if (snapshot_path.empty()) { return py::cast(nullptr); }
             return py::cast(snapshot_path);
           });
}

}  // namespace oneflow
