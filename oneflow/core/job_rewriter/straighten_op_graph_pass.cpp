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

#include <string>
#include "oneflow/core/auto_parallel/auto_memory.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/pass_util.h"
#include "oneflow/core/operator/op_conf.pb.h"
namespace oneflow {

namespace {

class StraightenOpGraphPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StraightenOpGraphPass);
  StraightenOpGraphPass() = default;
  ~StraightenOpGraphPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (ParseBooleanFromEnv("DISABLE_LOGICAL_STRAIGHTEN", false)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

Maybe<void> StraightenOpGraphPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  // TODO: use VLOG(3) here
  std::cout << "Straighten op graph is working!" << std::endl;
  std::vector<const OpNode*> ordered_op_nodes;
  auto_parallel::StraightenOpGraph(op_graph, &ordered_op_nodes);

  // Insert control edges for different placement
  HashMap<std::string, OperatorConf> mut_op_name2conf;
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  InsertCtrlEdgeInChain(ordered_op_nodes, IsReachable, &mut_op_name2conf);

  // TODO: Find the key point which screw up the DAG
  for (const auto& pair : mut_op_name2conf) { JUST(job_builder->MutOpOnlyOnce(pair.second)); }
  return Maybe<void>::Ok();
}

}  // anonymous namespace

REGISTER_JOB_PASS("StraightenOpGraphPass", StraightenOpGraphPass);

}  // namespace oneflow