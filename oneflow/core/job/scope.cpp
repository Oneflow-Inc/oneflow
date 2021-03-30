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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/interpreter.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/symbol_id_cache.h"

namespace oneflow {

Scope::Scope(const ScopeProto& scope_proto)
    : auto_increment_id_(0), symbol_id_(Error::SymbolIdUninitialized()), scope_proto_(scope_proto) {
  CHECK_OK(Init()) << scope_proto_.DebugString();
}

Scope::Scope(int64_t symbol_id, const ScopeProto& scope_proto)
    : auto_increment_id_(0), symbol_id_(symbol_id), scope_proto_(scope_proto) {}

Maybe<Scope> Scope::New(int64_t symbol_id, const ScopeProto& scope_proto) {
  auto* ptr = new Scope(symbol_id, scope_proto);
  std::shared_ptr<Scope> scope(ptr);
  JUST(scope->Init());
  return scope;
}

Maybe<void> Scope::Init() {
  {
    const auto& storage = *Global<symbol::Storage<JobDesc>>::Get();
    job_desc_ = JUST(storage.MaybeGetPtr(scope_proto_.job_desc_symbol_id()));
  }
  {
    const auto& storage = *Global<symbol::Storage<ParallelDesc>>::Get();
    device_parallel_desc_ =
        JUST(storage.MaybeGetPtr(scope_proto_.device_parallel_desc_symbol_id()));
    host_parallel_desc_ = JUST(storage.MaybeGetPtr(scope_proto_.host_parallel_desc_symbol_id()));
  }
  {
    const auto& storage = *Global<symbol::Storage<Scope>>::Get();
    if (scope_proto_.has_parent_scope_symbol_id()) {
      parent_scope_symbol_ = JUST(storage.MaybeGetPtr(scope_proto_.parent_scope_symbol_id()));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<const JobDesc*> Scope::job_desc() const {
  CHECK_NOTNULL_OR_RETURN(job_desc_.get());
  return job_desc_.get();
}

Maybe<int64_t> Scope::GetParallelDescSymbolId(const OperatorConf& op_conf) const {
  if (op_conf.device_tag() == "cpu" || IsCpuOnly(op_conf)) {
    return scope_proto_.host_parallel_desc_symbol_id();
  } else {
    return scope_proto_.device_parallel_desc_symbol_id();
  }
}

Maybe<const ParallelDesc&> Scope::GetParallelDesc(const OperatorConf& op_conf) const {
  if (op_conf.device_tag() == "cpu" || IsCpuOnly(op_conf)) {
    return *host_parallel_desc_;
  } else {
    return *device_parallel_desc_;
  }
}

const AttrValue& Scope::GetAttrValue(const std::string& attr_name) const {
  const auto& iter = scope_proto_.attr_name2attr_value().find(attr_name);
  if (iter != scope_proto_.attr_name2attr_value().end()) { return iter->second; }
  const auto& attr_name2attr_def = GlobalScopeConfigDef().attr_name2attr_def();
  const auto& def_iter = attr_name2attr_def.find(attr_name);
  CHECK(def_iter != attr_name2attr_def.end());
  return def_iter->second.default_val();
}

Maybe<cfg::ScopeProto> Scope::MakeChildScopeProto() const {
  auto child = std::make_shared<cfg::ScopeProto>(scope_proto_);
  child->set_parent_scope_symbol_id(JUST(symbol_id()));
  return child;
}

Maybe<int64_t> GenNewScopeSymbolId(
    int64_t old_scope_symbol_id,
    const std::function<void(std::shared_ptr<cfg::ScopeProto> new_scope)>& InitNewScopeProto) {
  CHECK_OR_RETURN(Global<symbol::Storage<Scope>>::Get()->Has(old_scope_symbol_id));
  const Scope& old_scope = Global<symbol::Storage<Scope>>::Get()->Get(old_scope_symbol_id);
  std::shared_ptr<cfg::ScopeProto> new_scope = JUST(old_scope.MakeChildScopeProto());
  InitNewScopeProto(new_scope);
  int64_t symbol_id = 0;
  JUST(LogicalInterpreter().Run([&](InstructionsBuilder* builder) -> Maybe<void> {
    symbol_id = JUST(builder->FindOrCreateSymbolId<cfg::ScopeProto>(*new_scope));
    return Maybe<void>::Ok();
  }));
  auto* id_cache = Global<symbol::IdCache<cfg::ScopeProto>>::Get();
  if (!id_cache->Has(*new_scope)) {
    JUST(
        id_cache->FindOrCreate(*new_scope, [&symbol_id]() -> Maybe<int64_t> { return symbol_id; }));
    ScopeProto scope_proto;
    new_scope->ToProto(&scope_proto);
    JUST(Global<symbol::Storage<Scope>>::Get()->TryAdd(symbol_id, scope_proto));
  }
  return symbol_id;
}

}  // namespace oneflow
