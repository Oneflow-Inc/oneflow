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
#include "oneflow/core/vm/string_symbol.h"

namespace py = pybind11;

namespace oneflow {

Maybe<StringSymbol> CreateStringSymbol(int64_t symbol_id, const std::string& data) {
  return std::make_shared<StringSymbol>(symbol_id, data);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<StringSymbol, std::shared_ptr<StringSymbol>>(m, "StringSymbol")
      .def(py::init([](int64_t symbol_id, const std::string& data) {
        return CreateStringSymbol(symbol_id, data).GetPtrOrThrow();
      }))
      .def_property_readonly("symbol_id",
                             [](const StringSymbol& x) { return x.symbol_id().GetOrThrow(); })
      .def_property_readonly("data", &StringSymbol::data);
}

}  // namespace oneflow
