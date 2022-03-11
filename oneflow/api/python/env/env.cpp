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
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/of_api_registry.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("CurrentResource", &oneflow::CurrentResource);
  m.def("EnvResource", &oneflow::EnvResource);
  m.def("EnableEagerEnvironment", &oneflow::EnableEagerEnvironment);

  m.def("IsEnvInited", &oneflow::IsEnvInited);
  m.def("InitEnv", &oneflow::InitEnv);
  m.def("DestroyEnv", &oneflow::DestroyEnv, py::call_guard<py::gil_scoped_release>());

  m.def("CurrentMachineId", &oneflow::CurrentMachineId);

  m.def("GetRank", &oneflow::GetRank);
  m.def("GetWorldSize", &oneflow::GetWorldSize);
  m.def("GetNodeSize", &oneflow::GetNodeSize);
  m.def("GetLocalRank", &oneflow::GetLocalRank);
  m.def("CudaGetDeviceCount", &oneflow::CudaGetDeviceCount);
  m.def("SetFLAGS_alsologtostderr", &oneflow::SetFLAGS_alsologtostderr);
  m.def("GetFLAGS_alsologtostderr", &oneflow::GetFLAGS_alsologtostderr);
  m.def("SetFLAGS_v", &oneflow::SetFLAGS_v);
  m.def("GetFLAGS_v", &oneflow::GetFLAGS_v);
}
