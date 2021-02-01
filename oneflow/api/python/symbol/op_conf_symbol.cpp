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
#include "oneflow/core/operator/op_conf_symbol.h"

namespace py = pybind11;

namespace oneflow {

Maybe<OperatorConfSymbol> CreateOpConfSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::OperatorConf>& symbol_conf) {
  return std::make_shared<OperatorConfSymbol>(symbol_id, symbol_conf);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<OperatorConfSymbol, std::shared_ptr<OperatorConfSymbol>>(m, "OpConfSymbol")
      .def(py::init([](int64_t symbol_id, const std::shared_ptr<cfg::OperatorConf>& symbol_conf) {
        return CreateOpConfSymbol(symbol_id, symbol_conf).GetPtrOrThrow();
      }))
      .def_property_readonly("symbol_id",
                             [](const OperatorConfSymbol& x) { return x.symbol_id().GetOrThrow(); })
      .def_property_readonly("data", &OperatorConfSymbol::data);
}

}  // namespace oneflow
