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
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace {

bool IsTransferNode(int32_t task_type) {
  return task_type == 12 || task_type == 13 || (48 <= task_type && task_type <= 64);
}

}  // anonymous namespace

// Drop down the maximum layer with the minimum layer form consumer
void TopoStruct::DropTributaryLayer(int32_t upper_bound) {
  if (upper_bound < TributaryLayer || TributaryLayer < 0) { TributaryLayer = upper_bound; }
}

// Should initialize the counter to be the number of out edges
// Compute maximum layer for tributaries
void TopoStruct::SpreadTributaryLayer(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  if (counter || MinLayer <= 0) { return; }
  int32_t producer_max_lay = 0;
  if (IfMainstem) {
    producer_max_lay = MinLayer - 1;
  } else {
    // On a tributary, the operator could be run later.
    producer_max_lay = TributaryLayer;
    // producer_max_lay = TributaryLayer - 1;
  }
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct[in];
    topo_struct_in.DropTributaryLayer(producer_max_lay);
    if (--topo_struct_in.counter == 0) {
      topo_struct_in.SpreadTributaryLayer(task_node2topo_struct);
    }
  });
  // Reduce counter to -1 to avoid visitting again
  counter--;
}

// Judge if this node is on the mainstem
// If so, judge it for its producer/upstream nodes
void TopoStruct::SpreadMainstem(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  // Skip it if this node is already judged.
  if (IfMainstem) { return; }
  CHECK(MinLayer >= 0) << "TopoStruct not initialized!";
  IfMainstem = true;
  // If I am in the mainstem, then all the children with (MinLayer >= my layer id - 1) would be
  // considered as in the mainstem
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct[in];
    if (topo_struct_in.MinLayer == MinLayer - 1) {
      topo_struct_in.SpreadTributaryLayer(task_node2topo_struct);
    }
  });
}

// The minimum computation distance from the beginning of this op to the next transfer
int32_t TopoStruct::GetMinDistance2Transfer(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  if (MinDistance2Transfer >= 0) { return MinDistance2Transfer; }
  // if this node is a transfer node
  if (IsTransferNode(node->GetTaskType())) {
    MinDistance2Transfer = 0;
    return MinDistance2Transfer;
  }
  MinDistance2Transfer = 1000000;
  node->ForEachNodeOnOutEdge([&](TaskNode* out) {
    MinDistance2Transfer =
        std::min(MinDistance2Transfer,
                 task_node2topo_struct[out].GetMinDistance2Transfer(task_node2topo_struct));
  });
  return ++MinDistance2Transfer;
}

// Find the mianstem of the task graph, then reduce the wait time for tributaries
void FindMainstem(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  // Find the maximum layer number
  int32_t max_MinLayer = -1;
  for (const auto& pair : task_node2topo_struct) {
    if (max_MinLayer < pair.second.MinLayer) { max_MinLayer = pair.second.MinLayer; }
  }
  // All the nodes with MinLayer>=mainstem_end_id would be considered as mainstem nodes
  int32_t mainstem_end_id = max_MinLayer - 4;
  for (auto& pair : task_node2topo_struct) {
    auto& topo_struct = pair.second;
    // Initialize the counter and Tributary Layer
    topo_struct.counter = pair.first->out_edges().size();
    topo_struct.TributaryLayer = max_MinLayer;
    // Find out all the nodes on the mainstem.
    if (topo_struct.MinLayer >= mainstem_end_id) {
      topo_struct.SpreadMainstem(task_node2topo_struct);
    }
  }

  for (auto& pair : task_node2topo_struct) {
    // Compute maximum layer for tributaries
    pair.second.SpreadTributaryLayer(task_node2topo_struct);
    // Set the MinDistance2Transfer for each topological structure
    pair.second.GetMinDistance2Transfer(task_node2topo_struct);
  }
}

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
