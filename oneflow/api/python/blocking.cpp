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

#include <glog/logging.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/thread_local_callback.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include <pybind11/functional.h>

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("blocking", m) {
  m.def("register_stack_info_callback", [](const std::function<std::string()>& Callback) {
    blocking::RegisterStackInfoCallback([Callback] {
      LOG(ERROR) << "[rank=" << std::to_string(GlobalProcessCtx::Rank()) << "]"
                 << " Blocking detected. Python stack:\n"
                 << Callback();
    });
  });
  m.def("clear_stack_info_callback", []() { blocking::ClearStackInfoCallback(); });
}

}  // namespace oneflow
