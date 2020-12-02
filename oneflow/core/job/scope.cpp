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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/vm/symbol_storage.h"

namespace oneflow {

Scope::Scope(const ScopeProto& scope_proto) : scope_proto_(scope_proto) {
  CHECK_OK(Init()) << scope_proto_.DebugString();
}

Maybe<void> Scope::Init() {
  {
    const auto& storage = *Global<vm::SymbolStorage<JobDesc>>::Get();
    job_desc_ = storage.GetPtr(scope_proto_.job_desc_symbol_id());
  }
  {
    const auto& storage = *Global<vm::SymbolStorage<ParallelDesc>>::Get();
    device_parallel_desc_ = storage.GetPtr(scope_proto_.device_parallel_desc_symbol_id());
    host_parallel_desc_ = storage.GetPtr(scope_proto_.host_parallel_desc_symbol_id());
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

}  // namespace oneflow
