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
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/device/cuda_util.h"

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

  py::class_<oneflow::EnvGlobalObjectsScope, std::shared_ptr<oneflow::EnvGlobalObjectsScope>>(
      m, "EnvContext")
      .def(py::init<const std::string&>())
      .def("SwitchToShuttingDownPhase", &SwitchToShuttingDownPhase,
           py::call_guard<py::gil_scoped_release>());

  m.def("CurrentMachineId", &CurrentMachineId);

  m.def("GetRank", &GetRank);
  m.def("GetWorldSize", &GetWorldSize);
  m.def("GetNodeSize", &GetNodeSize);
  m.def("GetLocalRank", &GetLocalRank);
  m.def("InitRdma", &InitRdma);
  m.def("RdmaInited", &RdmaInited);
  m.def("CudaGetDeviceCount", &CudaGetDeviceCount);
#ifdef WITH_CUDA
  m.def("GetCudaDeviceIndex", &GetCudaDeviceIndex);
  m.def("SetCudaDeviceIndex", &SetCudaDeviceIndex);
  m.def("CudaSynchronize", &CudaSynchronize);
#endif  // WITH_CUDA
  m.def("SetFLAGS_alsologtostderr", &SetFLAGS_alsologtostderr);
  m.def("GetFLAGS_alsologtostderr", &GetFLAGS_alsologtostderr);
  m.def("SetFLAGS_v", &SetFLAGS_v);
  m.def("GetFLAGS_v", &GetFLAGS_v);
  m.def("SetGraphLRVerbose", &SetGraphLRVerbose);
  m.def("GetGraphLRVerbose", &GetGraphLRVerbose);
  m.def("SetGraphDebugMaxPyStackDepth", &SetGraphDebugMaxPyStackDepth);
  m.def("GetGraphDebugMaxPyStackDepth", &GetGraphDebugMaxPyStackDepth);
  m.def("SetGraphDebugMode", &SetGraphDebugMode);
  m.def("GetGraphDebugMode", &GetGraphDebugMode);
}

}  // namespace oneflow
