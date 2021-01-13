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
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/op_node_signature.cfg.h"

namespace py = pybind11;

namespace oneflow {

Maybe<OpNodeSignatureDesc> CreateScopeSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::OpNodeSignature>& symbol_conf) {
  OpNodeSignature symbol_pb;
  symbol_conf->ToProto(&symbol_pb);
  return std::make_shared<OpNodeSignatureDesc>(symbol_id, symbol_pb);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<OpNodeSignatureDesc, std::shared_ptr<OpNodeSignatureDesc>>(m, "OpNodeSignatureSymbol")
      .def(
          py::init([](int64_t symbol_id, const std::shared_ptr<cfg::OpNodeSignature>& symbol_conf) {
            return CreateScopeSymbol(symbol_id, symbol_conf).GetPtrOrThrow();
          }))
      .def_property_readonly(
          "symbol_id", [](const OpNodeSignatureDesc& x) { return x.symbol_id().GetOrThrow(); })
      .def("data", &OpNodeSignatureDesc::cfg_op_node_signature);
}

}  // namespace oneflow
