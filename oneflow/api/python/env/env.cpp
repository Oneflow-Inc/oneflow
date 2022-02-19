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
#include "oneflow/api/python/env/env_api.h"

namespace py = pybind11;

namespace oneflow {
Maybe<void> EnableDTRStrategy(bool enable_dtr, size_t thres, int debug_level,
                              const std::string& heuristic) {
  CHECK_NOTNULL_OR_RETURN((Global<DTRConfig>::Get()));
  *Global<DTRConfig>::Get() = DTRConfig(enable_dtr, thres, debug_level, heuristic);
  return Maybe<void>::Ok();
}

Maybe<bool> CheckDTRStrategy() {
  CHECK_NOTNULL_OR_RETURN((Global<DTRConfig>::Get()));
  return Global<DTRConfig>::Get()->is_enabled;
}
}  // namespace oneflow

void ApiEnableDTRStrategy(bool enable_dtr, size_t thres, int debug_level,
                          const std::string& heuristic) {
  oneflow::EnableDTRStrategy(enable_dtr, thres, debug_level, heuristic).GetOrThrow();
}

bool ApiCheckDTRStrategy() {
   return oneflow::CheckDTRStrategy().GetOrThrow();
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("CurrentResource", &CurrentResource);
  m.def("EnvResource", &EnvResource);
  m.def("EnableEagerEnvironment", &EnableEagerEnvironment);
  m.def("EnableDTRStrategy", &ApiEnableDTRStrategy);
  m.def("CheckDTRStrategy", &ApiCheckDTRStrategy);

  m.def("IsEnvInited", &IsEnvInited);
  m.def("InitEnv", &InitEnv);
  m.def("DestroyEnv", &DestroyEnv, py::call_guard<py::gil_scoped_release>());

  m.def("CurrentMachineId", &CurrentMachineId);

  m.def("GetRank", &GetRank);
  m.def("GetWorldSize", &GetWorldSize);
  m.def("GetNodeSize", &GetNodeSize);
  m.def("GetLocalRank", &GetLocalRank);
  m.def("IsMultiClient", &IsMultiClient);
  m.def("SetIsMultiClient", &SetIsMultiClient);
  m.def("CudaGetDeviceCount", &CudaGetDeviceCount);
}
