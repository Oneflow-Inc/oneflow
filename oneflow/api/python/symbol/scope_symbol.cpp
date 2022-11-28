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
#include "oneflow/core/common/throw.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/scope.h"

namespace py = pybind11;

namespace oneflow {

Maybe<Scope> CreateScopeSymbol(int64_t symbol_id, const std::string& symbol_conf_str) {
  ScopeProto symbol_pb;
  if (!TxtString2PbMessage(symbol_conf_str, &symbol_pb)) {
    THROW(RuntimeError) << "symbol conf parse failed.\n" << symbol_conf_str;
  }
  return Scope::New(symbol_id, symbol_pb);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Scope, std::shared_ptr<Scope>>(m, "ScopeSymbol")
      .def(py::init([](int64_t symbol_id, const std::string& symbol_conf_str) {
        return CreateScopeSymbol(symbol_id, symbol_conf_str).GetPtrOrThrow();
      }))
      .def_property_readonly("symbol_id",
                             [](const Scope& x) {
                               if (!x.symbol_id().has_value()) {
                                 THROW(RuntimeError) << "symbol_id not initialized";
                               }
                               return CHECK_JUST(x.symbol_id());
                             })
      .def_property_readonly("_proto_str",
                             [](const Scope& x) { return PbMessage2TxtString(x.scope_proto()); })
      .def("auto_increment_id", &Scope::auto_increment_id)
      .def_property_readonly("session_id", &Scope::session_id)
      .def_property_readonly("job_desc_symbol", &Scope::job_desc_symbol)
      .def_property_readonly(
          "device_parallel_desc_symbol",
          [](const Scope& x) { return x.device_parallel_desc_symbol().shared_from_symbol(); })
      .def_property_readonly("parent_scope_symbol", &Scope::parent_scope_symbol)
      .def("MakeChildScopeProto", &Scope::MakeChildScopeProto);
}

}  // namespace oneflow
