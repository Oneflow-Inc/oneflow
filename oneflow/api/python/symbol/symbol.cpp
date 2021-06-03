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
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_node_signature.cfg.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/vm/string_symbol.h"

namespace py = pybind11;

namespace oneflow {

namespace {

template<typename SymbolConfT>
bool ApiHasSymbol(const SymbolConfT& symbol_conf) {
  return HasSymbol(symbol_conf).GetOrThrow();
}

template<typename SymbolConfT, typename SymbolPbT, typename SymbolT>
void ApiAddSymbol(int64_t symbol_id, const SymbolConfT& symbol_conf) {
  return AddSymbol<SymbolConfT, SymbolPbT, SymbolT>(symbol_id, symbol_conf).GetOrThrow();
}

template<typename SymbolConfT, typename SymbolT>
std::shared_ptr<SymbolT> ApiGetSymbol(const SymbolConfT& symbol_conf) {
  return GetSymbol<SymbolConfT, SymbolT>(symbol_conf).GetPtrOrThrow();
}

template<typename SymbolConfT, typename SymbolT>
std::shared_ptr<SymbolT> ApiGetSymbolById(int64_t symbol_id) {
  return GetSymbol<SymbolConfT, SymbolT>(symbol_id).GetPtrOrThrow();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("HasPlacementSymbol", &ApiHasSymbol<cfg::ParallelConf>);
  m.def("AddPlacementSymbol", &ApiAddSymbol<cfg::ParallelConf, ParallelConf, ParallelDesc>);
  m.def("GetPlacementSymbol", &ApiGetSymbol<cfg::ParallelConf, ParallelDesc>);
  m.def("GetPlacementSymbol", &ApiGetSymbolById<cfg::ParallelConf, ParallelDesc>);

  m.def("HasJobConfSymbol", &ApiHasSymbol<cfg::JobConfigProto>);
  m.def("AddJobConfSymbol", &ApiAddSymbol<cfg::JobConfigProto, JobConfigProto, JobDesc>);
  m.def("GetJobConfSymbol", &ApiGetSymbol<cfg::JobConfigProto, JobDesc>);
  m.def("GetJobConfSymbol", &ApiGetSymbolById<cfg::JobConfigProto, JobDesc>);

  m.def("HasScopeSymbol", &ApiHasSymbol<cfg::ScopeProto>);
  m.def("AddScopeSymbol", &ApiAddSymbol<cfg::ScopeProto, ScopeProto, Scope>);
  m.def("GetScopeSymbol", &ApiGetSymbol<cfg::ScopeProto, Scope>);
  m.def("GetScopeSymbol", &ApiGetSymbolById<cfg::ScopeProto, Scope>);

  m.def("HasOpNodeSignatureSymbol", &ApiHasSymbol<cfg::OpNodeSignature>);
  m.def("AddOpNodeSignatureSymbol",
        &ApiAddSymbol<cfg::OpNodeSignature, OpNodeSignature, OpNodeSignatureDesc>);
  m.def("GetOpNodeSignatureSymbol", &ApiGetSymbol<cfg::OpNodeSignature, OpNodeSignatureDesc>);
  m.def("GetOpNodeSignatureSymbol", &ApiGetSymbolById<cfg::OpNodeSignature, OpNodeSignatureDesc>);
}

}  // namespace oneflow
