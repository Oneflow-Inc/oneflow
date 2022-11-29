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
#include "oneflow/core/odb/odb.h"

ONEFLOW_API_PYBIND11_MODULE("odb", m) {
  using namespace oneflow;
  namespace py = pybind11;

  m.def("SetNormalBreakpoint", []() { odb::SetThreadTypeBreakpoint(odb::kNormalThreadType); });
  m.def("SetSchedulerBreakpoint",
        []() { odb::SetThreadTypeBreakpoint(odb::kSchedulerThreadType); });
  m.def("SetWorkerBreakpoint", []() { odb::SetThreadTypeBreakpoint(odb::kWorkerThreadType); });
  m.def("SetAllBreakpoints", []() {
    odb::SetThreadTypeBreakpoint(odb::kNormalThreadType);
    odb::SetThreadTypeBreakpoint(odb::kSchedulerThreadType);
    odb::SetThreadTypeBreakpoint(odb::kWorkerThreadType);
  });

  m.def("ClearNormalBreakpoint", []() { odb::ClearThreadTypeBreakpoint(odb::kNormalThreadType); });
  m.def("ClearSchedulerBreakpoint",
        []() { odb::ClearThreadTypeBreakpoint(odb::kSchedulerThreadType); });
  m.def("ClearWorkerBreakpoint", []() { odb::ClearThreadTypeBreakpoint(odb::kWorkerThreadType); });
  m.def("ClearAllBreakpoints", []() {
    odb::ClearThreadTypeBreakpoint(odb::kNormalThreadType);
    odb::ClearThreadTypeBreakpoint(odb::kSchedulerThreadType);
    odb::ClearThreadTypeBreakpoint(odb::kWorkerThreadType);
  });

  m.def("StopVMScheduler", []() { return odb::StopVMThread<odb::kSchedulerThreadType>(); });
  m.def("RestartVMScheduler", []() { return odb::RestartVMThread<odb::kSchedulerThreadType>(); });
  m.def("StopVMWorker", []() { return odb::StopVMThread<odb::kWorkerThreadType>(); });
  m.def("RestartVMWorker", []() { return odb::RestartVMThread<odb::kWorkerThreadType>(); });
}
