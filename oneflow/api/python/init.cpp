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
#include <pybind11/functional.h>
#include "oneflow/core/common/global.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/thread/glog_failure_function.h"
#include "oneflow/core/job/api_helper.h"

namespace py = pybind11;

namespace oneflow {

PYBIND11_MODULE(oneflow_api, m) {
  py::register_exception<MainThreadPanic>(m, "MainThreadPanic");
  m.def("EagerExecutionEnabled", []() { return EagerExecutionEnabled(); });
  m.def("SetPanicCallback", [](const GlogFailureFunction::py_failure_callback& f) {
    Global<GlogFailureFunction>::Get()->SetCallback(f);
  });
  m.def("StartGlobalSession", []() {
    std::string error_str;
    StartGlobalSession().GetDataAndSerializedErrorProto(&error_str);
    return error_str;
  });
}

}  // namespace oneflow
