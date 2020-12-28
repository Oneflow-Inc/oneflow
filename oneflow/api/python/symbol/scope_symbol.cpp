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
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"

namespace py = pybind11;

namespace oneflow {

Maybe<Scope> CreateScopeSymbol(int64_t symbol_id,
                               const std::shared_ptr<cfg::ScopeProto>& symbol_conf) {
  ScopeProto symbol_pb;
  symbol_conf->ToProto(&symbol_pb);
  return Scope::New(symbol_id, symbol_pb);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Scope, std::shared_ptr<Scope>>(m, "ScopeSymbol")
      .def(py::init([](int64_t symbol_id, const std::shared_ptr<cfg::ScopeProto>& symbol_conf) {
        return CreateScopeSymbol(symbol_id, symbol_conf).GetPtrOrThrow();
      }))
      .def_property_readonly("symbol_id", [](const Scope& x) { return x.symbol_id().GetOrThrow(); })
      .def("auto_increment_id", &Scope::auto_increment_id)
      .def_property_readonly("session_id", &Scope::session_id)
      .def_property_readonly("session_id", &Scope::session_id)
      .def_property_readonly("job_desc_symbol", &Scope::job_desc_symbol)
      .def_property_readonly("device_parallel_desc_symbol", &Scope::device_parallel_desc_symbol)
      .def_property_readonly("parent_scope_symbol", &Scope::parent_scope_symbol)
      .def("MakeChildScopeProto",
           [](const Scope& scope) { return scope.MakeChildScopeProto().GetOrThrow(); });
}

}  // namespace oneflow
