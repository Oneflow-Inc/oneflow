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
#include "oneflow/core/job_rewriter/straighten_nodes.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/auto_parallel/sbp_constructor.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

Maybe<void> StraightenNodes(const OpGraph& op_graph, Job* job) {
  // Not allowed two-step boxing and disable checking for debugging
  if (ParseBooleanFromEnv("ONEFLOW_RANDOM_STRAIGHTEN_NODES", false)) { return Maybe<void>::Ok(); }
  // test debug
  if (GlobalProcessCtx::Rank() == 0) { std::cout << "Start straightening operators" << std::endl; }
  auto_parallel::SbpConstructor sbp_constructor(op_graph, job, /*take_curr_sbp=*/true);
  sbp_constructor.ExposeCtrlEdges();
  // Add control edge
  JobBuilder job_builder(job);
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  // Add a control edge from the previous node to this node
  auto add_control_edge = [&](OpNode* previous_node, OpNode* this_node) -> Maybe<void> {
    const auto& previous_name = previous_node->op().op_conf().name();
    const auto& this_conf = this_node->op().op_conf();
    if (!IsReachable(previous_name, this_conf.name())) {
      OperatorConf mutable_consumer_op_conf(this_conf);
      mutable_consumer_op_conf.add_ctrl_in_op_name(previous_name);
      JUST(job_builder.MutOpOnlyOnce(mutable_consumer_op_conf));
    }
    return Maybe<void>::Ok();
  };
  JUST(sbp_constructor.StraightenNodes(add_control_edge));

  return Maybe<void>::Ok();
}

}  // namespace oneflow
