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
#include <string>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/api/python/framework/framework_api.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("RegisterGlobalForeignCallback", &RegisterGlobalForeignCallback);
  m.def("DestroyGlobalForeignCallback", &DestroyGlobalForeignCallback);
  m.def("RegisterGlobalWatcher", &RegisterGlobalWatcher);
  m.def("LaunchJob", &LaunchJob, py::call_guard<py::gil_scoped_release>());

  m.def("GetSerializedInterUserJobInfo",
        []() { return py::bytes(GetSerializedInterUserJobInfo()); });
  m.def("GetSerializedJobSet", []() { return py::bytes(GetSerializedJobSet()); });
  m.def("GetSerializedStructureGraph", &GetSerializedStructureGraph /* a prototxt saved to file*/);
  m.def("GetSerializedCurrentJob", []() { return py::bytes(GetSerializedCurrentJob()); });

  m.def("GetFunctionConfigDef", &GetFunctionConfigDef);
  m.def("GetScopeConfigDef", &GetScopeConfigDef);
  m.def("GetMachine2DeviceIdListOFRecordFromParallelConf",
        &GetMachine2DeviceIdListOFRecordFromParallelConf);

  m.def("LoadSavedModel", &LoadSavedModel);

  m.def("EagerExecutionEnabled", []() { return oneflow::EagerExecutionEnabled(); });
  m.def("LoadLibraryNow", &LoadLibraryNow);
}
