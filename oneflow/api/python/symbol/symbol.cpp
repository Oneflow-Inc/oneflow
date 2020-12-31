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
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job/scope.pb.h"

namespace py = pybind11;

namespace oneflow {

namespace {

template<typename SymbolConfT>
bool ApiHasSymbol(const SymbolConfT& symbol_conf) {
  const auto& id_cache = *Global<symbol::IdCache<SymbolConfT>>::Get();
  return id_cache.Has(symbol_conf);
}

template<typename SymbolConfT, typename SymbolT>
Maybe<SymbolT> GetSymbol(const SymbolConfT& symbol_conf) {
  const auto& id_cache = *Global<symbol::IdCache<SymbolConfT>>::Get();
  const auto& symbol_storage = *Global<symbol::Storage<SymbolT>>::Get();
  int64_t symbol_id = JUST(id_cache.Get(symbol_conf));
  const auto& ptr = JUST(symbol_storage.MaybeGetPtr(symbol_id));
  JUST(ptr->symbol_id());
  return ptr;
}

// TODO(hanbibin): the second template arg will be moved after symbol_storage is refactored
template<typename SymbolConfT, typename SymbolPbT, typename SymbolT>
Maybe<void> AddSymbol(int64_t symbol_id, const SymbolConfT& symbol_conf) {
  SymbolPbT symbol_pb;
  symbol_conf.ToProto(&symbol_pb);
  JUST(Global<symbol::Storage<SymbolT>>::Get()->Add(symbol_id, symbol_pb));
  auto* id_cache = Global<symbol::IdCache<SymbolConfT>>::Get();
  CHECK_OR_RETURN(!id_cache->Has(symbol_conf));
  JUST(id_cache->FindOrCreate(symbol_conf, [&symbol_id]() -> Maybe<int64_t> { return symbol_id; }));
  return Maybe<void>::Ok();
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
Maybe<SymbolT> GetSymbol(int64_t symbol_id) {
  const auto& symbol_storage = *Global<symbol::Storage<SymbolT>>::Get();
  const auto& ptr = JUST(symbol_storage.MaybeGetPtr(symbol_id));
  JUST(ptr->symbol_id());
  return ptr;
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
}

}  // namespace oneflow
