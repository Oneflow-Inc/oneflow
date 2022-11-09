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
#include "oneflow/core/framework/scope_util.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("GetCurrentScope", &GetCurrentScope);
  m.def("MakeInitialScope",
        [](const std::string& job_conf_str, Symbol<ParallelDesc> placement,
           bool is_local) -> Maybe<Scope> {
          JobConfigProto job_conf;
          CHECK_OR_RETURN(TxtString2PbMessage(job_conf_str, &job_conf)) << "job conf parse failed";
          return MakeInitialScope(job_conf, placement, is_local);
        });
  m.def("InitGlobalScopeStack", &InitThreadLocalScopeStack);

  m.def("GlobalScopeStackPush", &ThreadLocalScopeStackPush);
  m.def("GlobalScopeStackPop", &ThreadLocalScopeStackPop);
}

}  // namespace oneflow
