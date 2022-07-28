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
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

class PruneDanglingConstantPass final : public JobPass {
 public:
  PruneDanglingConstantPass() = default;
  ~PruneDanglingConstantPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> PruneDanglingConstantPass::Apply(const OpGraph& op_graph,
                                             JobBuilder* job_builder) const {
  std::vector<OperatorConf> delete_ops;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (!IsUserOpWithTypeName(op_node->op().op_conf(), "constant")
        || op_node->out_edges().size() > 0) {
      return;
    }
    delete_ops.emplace_back(op_node->op().op_conf());
  });

  job_builder->DelOps(delete_ops);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PruneDanglingConstantPass", PruneDanglingConstantPass);

}  // namespace oneflow
