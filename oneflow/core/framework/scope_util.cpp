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
                              bool is_mirrored) {
  std::shared_ptr<Scope> scope;
  JUST(PhysicalRun([&scope, &job_conf, placement,
                    is_mirrored](InstructionsBuilder* builder) -> Maybe<void> {
    int64_t session_id = JUST(GetDefaultSessionId());
    scope =
        JUST(builder->BuildInitialScopeWithPlacement(session_id, job_conf, placement, is_mirrored));
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

}  // namespace oneflow
