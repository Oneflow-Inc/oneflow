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
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/core/common/maybe.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<OperatorConfSymbol, std::shared_ptr<OperatorConfSymbol>>(m, "OpConfSymbol")
      .def_property_readonly("symbol_id",
                             [](const OperatorConfSymbol& x) {
                               if (!x.symbol_id().has_value()) {
                                 THROW(RuntimeError) << "symbol_id not initialized";
                               }
                               return CHECK_JUST(x.symbol_id());
                             })
      .def_property_readonly("data", &OperatorConfSymbol::data);
}

}  // namespace oneflow
