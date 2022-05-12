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
  return Maybe<void>::Ok();
  if (ParseBooleanFromEnv("ONEFLOW_RANDOM_STRAIGHTEN_NODES", false)) { return Maybe<void>::Ok(); }
  // test debug
  if (GlobalProcessCtx::Rank() == 0) { std::cout << "Start straightening operators" << std::endl; }
  auto_parallel::SbpConstructor sbp_constructor(op_graph, job, /*take_curr_sbp=*/true);
  sbp_constructor.ExposeCtrlEdges();
  // Add control edge
  JobBuilder job_builder(job);
  // Judge whether we can set a control edge from source node to destination node
  // We set up this function from task_graph.cpp:ForEachOpGraphNecessaryCtrlEdge()
  auto IsOpGraphDataReachable = op_graph.MakePredicatorIsReachable();
  auto able_to_add_control_edge = [&](OpNode* src, OpNode* dst) {
    if (IsOpGraphDataReachable(dst, src)) { return false; }
    if (!IsOpGraphDataReachable(src, dst)) {
      if (dst->parallel_desc().parallel_num() != src->parallel_desc().parallel_num()) {
        return false;
      }
      const Shape* src_time_shape = CHECK_JUST(src->op().GetOpTimeShape()).get();
      const Shape* dst_time_shape = CHECK_JUST(dst->op().GetInputBlobFastestTimeShape()).get();
      if (dst_time_shape == nullptr) {
        dst_time_shape = CHECK_JUST(dst->op().GetOpTimeShape()).get();
      }
      if (src_time_shape->elem_cnt() != dst_time_shape->elem_cnt()) { return false; }
    }
    return true;
  };
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  // Add a control edge from the previous node to this node
  auto add_control_edge = [&](OpNode* previous_node, OpNode* this_node) -> Maybe<void> {
    const auto& previous_name = previous_node->op().op_conf().name();
    const auto& this_conf = this_node->op().op_conf();
    if (!IsReachable(previous_name, this_conf.name())
        && able_to_add_control_edge(previous_node, this_node)) {
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
