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
#include <list>
#include "oneflow/core/framework/scope_util.h"

#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {

namespace {

Maybe<Scope> MakeDefaultScope() {
  JobConfigProto config_proto;
  config_proto.mutable_predict_conf();
  config_proto.set_job_name("");
  return MakeScope(config_proto, *JUST(Device::New("cpu")));
}

std::list<std::shared_ptr<Scope>>* ThreadLocalScopeStack() {
  thread_local static std::list<std::shared_ptr<Scope>> scope_stack{CHECK_JUST(MakeDefaultScope())};
  return &scope_stack;
}

}  // namespace

Maybe<Scope> MakeScope(const JobConfigProto& config_proto, const Device& device) {
  std::shared_ptr<Scope> scope;
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    int64_t session_id = JUST(GetDefaultSessionId());
    std::string device_tag = "cpu";
    std::string machine_ids = "0";
    std::string device_ids = "0";
    if (device.type() != "cpu") {
      device_tag = device.type();
      device_ids = std::to_string(device.device_id());
    }
    scope = JUST(builder->BuildInitialScope(session_id, config_proto, device_tag,
                                            {machine_ids + ":" + device_ids}, nullptr, false));
    return Maybe<void>::Ok();
  }));
  return scope;
}

Maybe<Scope> MakeInitialScope(const JobConfigProto& job_conf, Symbol<ParallelDesc> placement,
                              bool is_local) {
  std::shared_ptr<Scope> scope;
  JUST(PhysicalRun([&scope, &job_conf, placement,
                    is_local](InstructionsBuilder* builder) -> Maybe<void> {
    int64_t session_id = JUST(GetDefaultSessionId());
    scope =
        JUST(builder->BuildInitialScopeWithPlacement(session_id, job_conf, placement, is_local));
    return Maybe<void>::Ok();
  }));
  return scope;
}

Maybe<Scope> GetCurrentScope() {
  auto* scope_stack = ThreadLocalScopeStack();
  CHECK_GT_OR_RETURN(scope_stack->size(), 0);
  return scope_stack->back();
}

Maybe<void> InitThreadLocalScopeStack(const std::shared_ptr<Scope>& scope) {
  auto* scope_stack = ThreadLocalScopeStack();
  scope_stack->clear();
  scope_stack->emplace_back(scope);
  return Maybe<void>::Ok();
}

Maybe<void> ThreadLocalScopeStackPush(const std::shared_ptr<Scope>& scope) {
  auto* scope_stack = ThreadLocalScopeStack();
  scope_stack->emplace_back(scope);
  return Maybe<void>::Ok();
}

Maybe<void> ThreadLocalScopeStackPop() {
  auto* scope_stack = ThreadLocalScopeStack();
  scope_stack->pop_back();
  return Maybe<void>::Ok();
}

BackwardPassScopeGuard::BackwardPassScopeGuard() {
  if (LazyMode::is_enabled()) {
    const auto& scope = CHECK_JUST(GetCurrentScope());
    if (scope) {
      backward_pass_scope_ = CHECK_JUST(FindOrCreateBackwardPassScope(scope));
      CHECK_JUST(ThreadLocalScopeStackPush(backward_pass_scope_));
    }
  }
}

BackwardPassScopeGuard::BackwardPassScopeGuard(const std::shared_ptr<Scope>& scope) {
  if (scope && LazyMode::is_enabled()) {
    backward_pass_scope_ = CHECK_JUST(FindOrCreateBackwardPassScope(scope));
    CHECK_JUST(ThreadLocalScopeStackPush(backward_pass_scope_));
  }
}

BackwardPassScopeGuard::~BackwardPassScopeGuard() {
  if (backward_pass_scope_) { CHECK_JUST(ThreadLocalScopeStackPop()); }
}

class BackwardPassScopeStorage {
 public:
  std::mutex mutex;

  static BackwardPassScopeStorage* Global() {
    static BackwardPassScopeStorage instance;
    return &instance;
  }
  HashMap<int64_t, std::shared_ptr<Scope>>& get() { return scopes_; }

 private:
  HashMap<int64_t, std::shared_ptr<Scope>> scopes_;
};

extern const std::string kBackwardPass;
Maybe<Scope> FindOrCreateBackwardPassScope(const std::shared_ptr<Scope>& scope) {
  auto* storage = BackwardPassScopeStorage::Global();
  auto& scopes = storage->get();
  std::lock_guard<std::mutex> lock(storage->mutex);
  auto it = scopes.find(JUST(scope->symbol_id()));
  if (it != scopes.end()) { return it->second; }
  auto scope_proto = JUST((scope->MakeChildScopeProto()));
  scope_proto->set_calculation_pass_name(kBackwardPass);
  std::shared_ptr<Scope> backward_pass_scope;
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    backward_pass_scope = JUST(builder->GetScopeSymbol(*scope_proto));
    return Maybe<void>::Ok();
  }));
  scopes.emplace(JUST(scope->symbol_id()), backward_pass_scope);
  return backward_pass_scope;
}

void ClearAllBackwardPassScope() {
  auto* storage = BackwardPassScopeStorage::Global();
  std::lock_guard<std::mutex> lock(storage->mutex);
  storage->get().clear();
}

}  // namespace oneflow
