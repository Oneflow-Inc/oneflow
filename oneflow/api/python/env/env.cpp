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
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/job/graph_scope_vars.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/mem_util.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif  // WITH_CUDA

namespace py = pybind11;

namespace oneflow {

#ifdef WITH_CUDA

void RegisterCudaDeviceProperties(py::module& m) {
  py::class_<cudaDeviceProp>(m, "_CudaDeviceProperties", py::module_local())
      .def(py::init<>())
      .def_readonly("name", &cudaDeviceProp::name)
      .def_readonly("major", &cudaDeviceProp::major)
      .def_readonly("minor", &cudaDeviceProp::minor)
      .def_readonly("is_multi_gpu_board", &cudaDeviceProp::isMultiGpuBoard)
      .def_readonly("is_integrated", &cudaDeviceProp::integrated)
      .def_readonly("multi_processor_count", &cudaDeviceProp::multiProcessorCount)
      .def_readonly("total_memory", &cudaDeviceProp::totalGlobalMem)
      .def("__repr__", [](const cudaDeviceProp& prop) {
        std::ostringstream stream;
        stream << "_CudaDeviceProperties(name='" << prop.name << "', major=" << prop.major
               << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount << ")";
        return stream.str();
      });
}

#endif  // WITH_CUDA

Maybe<void> SwitchToShuttingDownPhase(EnvGlobalObjectsScope* env, bool is_normal_exit) {
  JUST(env->init_is_normal_exit(is_normal_exit));
  SetShuttingDown(true);
  if (is_normal_exit) {
    JUST(vm::ClusterSync());
    auto* vm = JUST(SingletonMaybe<VirtualMachine>());
    JUST(vm->CloseVMThreads());
  }
  return Maybe<void>::Ok();
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("CurrentResource", &CurrentResource);
  m.def("EnvResource", &EnvResource);

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
  m.def("InitRDMA", &InitRDMA);
  m.def("RDMAIsInitialized", &RDMAIsInitialized);
  m.def("DestoryRDMA", &DestoryRDMA);
  m.def("CudaGetDeviceCount", &CudaGetDeviceCount);
  m.def("EmptyCache", &EmptyCache);
#ifdef WITH_CUDA
  RegisterCudaDeviceProperties(m);
  m.def("GetCudaDeviceIndex", &GetCudaDeviceIndex);
  m.def("SetCudaDeviceIndex", &SetCudaDeviceIndex);
  m.def("CudaSynchronize", &CudaSynchronize);
  m.def("GetCUDAMemoryUsed", &GetCUDAMemoryUsed);
  m.def("GetCPUMemoryUsed", &GetCPUMemoryUsed);
  m.def(
      "_get_device_properties",
      [](int device) -> cudaDeviceProp* { return GetDeviceProperties(device); },
      py::return_value_policy::reference);
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
  m.def("SetGraphDebugOnlyUserPyStack", &SetGraphDebugOnlyUserPyStack);
  m.def("GetGraphDebugOnlyUserPyStack", &GetGraphDebugOnlyUserPyStack);
  m.def("InitPythonPathsToBeKeptAndFilteredForDebugging",
        &InitPythonPathsToBeKeptAndFilteredForDebugging);
}

}  // namespace oneflow
