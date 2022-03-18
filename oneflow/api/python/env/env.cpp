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
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/framework/shut_down_util.h"

namespace py = pybind11;

namespace oneflow {

Maybe<void> SwitchToShuttingDownPhase(EnvGlobalObjectsScope* env, bool is_normal_exit) {
  if (is_normal_exit) {
    JUST(vm::ClusterSync());
    auto* vm = JUST(GlobalMaybe<VirtualMachine>());
    JUST(vm->CloseVMThreads());
  }
  JUST(env->init_is_normal_exit(is_normal_exit));
  SetShuttingDown(true);
  return Maybe<void>::Ok();
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("CurrentResource", &CurrentResource);
  m.def("EnvResource", &EnvResource);
  m.def("EnableEagerEnvironment", &EnableEagerEnvironment);

  py::class_<EnvGlobalObjectsScope, std::shared_ptr<EnvGlobalObjectsScope>>(m, "Env")
      .def(py::init([](const std::string& env_proto_str) {
        return CreateEnv(env_proto_str).GetPtrOrThrow();
      }))
      .def(
          "SwitchToShuttingDownPhase",
          [](EnvGlobalObjectsScope* env, bool is_normal_exit) {
            SwitchToShuttingDownPhase(env, is_normal_exit).GetOrThrow();
          },
          py::call_guard<py::gil_scoped_release>());

  m.def("CurrentMachineId", &CurrentMachineId);

  m.def("GetRank", &GetRank);
  m.def("GetWorldSize", &GetWorldSize);
  m.def("GetNodeSize", &GetNodeSize);
  m.def("GetLocalRank", &GetLocalRank);
  m.def("CudaGetDeviceCount", &CudaGetDeviceCount);
  m.def("SetFLAGS_alsologtostderr", &SetFLAGS_alsologtostderr);
  m.def("GetFLAGS_alsologtostderr", &GetFLAGS_alsologtostderr);
  m.def("SetFLAGS_v", &SetFLAGS_v);
  m.def("GetFLAGS_v", &GetFLAGS_v);
  m.def("SetGraphLRVerbose", &SetGraphLRVerbose);
  m.def("GetGraphLRVerbose", &GetGraphLRVerbose);
}

}  // namespace oneflow
