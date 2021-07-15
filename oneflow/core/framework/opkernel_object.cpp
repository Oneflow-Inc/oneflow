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
#include "oneflow/core/framework/opkernel_object.h"
#include "oneflow/core/framework/symbol_storage_util.h"

namespace oneflow {

namespace compatible_py {

namespace {

std::shared_ptr<Scope> GetScopeSymbol(const std::shared_ptr<cfg::OperatorConf>& op_conf) {
  CHECK(op_conf->has_scope_symbol_id());
  return CHECK_JUST(GetSymbol<cfg::ScopeProto, Scope>(op_conf->scope_symbol_id()));
}

std::shared_ptr<ParallelDesc> GetOpParallelSymbol(
    const std::shared_ptr<cfg::OperatorConf>& op_conf) {
  CHECK(op_conf->has_scope_symbol_id());
  const auto& scope = Global<symbol::Storage<Scope>>::Get()->Get(op_conf->scope_symbol_id());
  OperatorConf pb_op_conf;
  op_conf->ToProto(&pb_op_conf);
  int64_t parallel_desc_symbol_id = CHECK_JUST(scope.GetParallelDescSymbolId(pb_op_conf));
  return CHECK_JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_symbol_id));
}

}  // namespace

OpKernelObject::OpKernelObject(int64_t object_id, const std::shared_ptr<cfg::OperatorConf>& op_conf,
                               const std::function<void(Object*)>& release)
    : Object(object_id, GetOpParallelSymbol(op_conf)),
      op_conf_(op_conf),
      scope_symbol_(GetScopeSymbol(op_conf)) {
  release_.emplace_back(release);
}

void OpKernelObject::ForceReleaseAll() {
  for (const auto& release : release_) { release(this); }
}

}  // namespace compatible_py

}  // namespace oneflow
