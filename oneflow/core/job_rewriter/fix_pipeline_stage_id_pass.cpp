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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

class FixPipelineStageIdPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FixPipelineStageIdPass);
  FixPipelineStageIdPass() = default;
  ~FixPipelineStageIdPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

Maybe<Scope> NewScopeWithStageId(const std::shared_ptr<Scope>& old_scope, int64_t stage_id) {
  std::shared_ptr<Scope> new_scope;
  const auto SetScopeProto = [stage_id](const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  auto* attr_map = scope_proto->mutable_attr_name2attr_value();
  (*attr_map)[""]
  const auto& iter = scope_proto_.attr_name2attr_value().find(attr_name);
  if (iter != scope_proto_.attr_name2attr_value().end()) { return iter->second; }
  const auto& attr_name2attr_def = GlobalScopeConfigDef().attr_name2attr_def();
  const auto& def_iter = attr_name2attr_def.find(attr_name);
  CHECK(def_iter != attr_name2attr_def.end());
  return def_iter->second.default_val();

    scope_proto->mutable_opt_mirrored_parallel_conf()->mutable_mirrored_parallel();
  };

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    new_scope = JUST(builder->BuildScopeWithNewParallelConf(old_scope, parallel_conf));
    return Maybe<void>::Ok();
  }));
  // NOTE(chengcheng): need sync vm for get scope right now
  JUST(vm::CurrentRankSync());
  CHECK_OR_RETURN(new_scope);
  return new_scope;
}


Maybe<void> FixPipelineStageIdPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {

}
}  // namespace

REGISTER_JOB_PASS("FixPipelineStageIdPass", FixPipelineStageIdPass);

}  // namespace oneflow
