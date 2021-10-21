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
  m.def("GetCurrentScope", []() { return GetCurrentScope().GetPtrOrThrow(); });
  m.def("InitGlobalScopeStack", [](const std::shared_ptr<Scope>& scope) {
    return InitThreadLocalScopeStack(scope).GetOrThrow();
  });

  m.def("GlobalScopeStackPush", [](const std::shared_ptr<Scope>& scope) {
    return ThreadLocalScopeStackPush(scope).GetOrThrow();
  });
  m.def("GlobalScopeStackPop", []() { return ThreadLocalScopeStackPop().GetOrThrow(); });
}

}  // namespace oneflow
