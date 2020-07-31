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
#include "oneflow/core/graph/sharable_mem_block_graph.h"
#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/graph/inplace_regst_graph.h"

namespace oneflow {

namespace {

bool IsConsumersAndProducerInSameChain(const RegstDescProto& regst_desc,
                                       const PlanTaskGraph& plan_task_graph) {
  auto ChainId4TaskId = [&](int64_t task_id) {
    return plan_task_graph.TaskProto4TaskId(task_id)->task_set_info().chain_id();
  };
  int64_t producer_chain_id = ChainId4TaskId(regst_desc.producer_task_id());
  for (int64_t consumer_task_id : regst_desc.consumer_task_id()) {
    if (ChainId4TaskId(consumer_task_id) != producer_chain_id) { return false; }
  }
  return true;
}
void ForEachInplacedRegstDescs(
    const HashSet<const RegstDescProto*> regst_desc,
    const std::function<void(const HashSet<const RegstDescProto*>&)>& Handler) {
  InplaceRegstGraph inplace_gph(regst_desc);
  inplace_gph.ForEachConnectedComponent([&](const HashSet<const InplaceRegstNode*>& nodes) {
    if (nodes.size() == 1) { return; }
    HashSet<const RegstDescProto*> regst_descs;
    for (const auto* node : nodes) { CHECK(regst_descs.emplace(node->regst_desc()).second); }
    Handler(regst_descs);
  });
}

}  // namespace

SharableMemBlockNode::SharableMemBlockNode(int64_t chain_id,
                                           const HashSet<const RegstDescProto*>& regst_descs)
    : chain_id_(chain_id), regst_descs_(regst_descs.begin(), regst_descs.end()) {}

SharableMemBlockGraph::SharableMemBlockGraph(
    const PlanTaskGraph& plan_task_gph,
    const std::function<bool(const RegstDescProto&)>& IsSharable) {
  HashMap<int64_t, HashSet<const RegstDescProto*>> chain_id2regst_descs;
  for (const TaskProto& task : plan_task_gph.plan().task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      if (IsConsumersAndProducerInSameChain(pair.second, plan_task_gph)
          && IsSharable(pair.second)) {
        CHECK(chain_id2regst_descs[task.task_set_info().chain_id()].emplace(&pair.second).second);
      }
    }
  }
  for (const auto& pair : chain_id2regst_descs) {
    HashMap<const RegstDescProto*, SharableMemBlockNode*> regst_desc2node;
    for (const auto* regst_desc : pair.second) {
      auto* node = new SharableMemBlockNode(pair.first, {regst_desc});
      AddAllocatedNode(node);
      CHECK(regst_desc2node.emplace(regst_desc, node).second);
    }
    ForEachInplacedRegstDescs(pair.second, [&](const HashSet<const RegstDescProto*>& regst_descs) {
      auto* parent = new SharableMemBlockNode(pair.first, regst_descs);
      AddAllocatedNode(parent);
      for (const RegstDescProto* regst_desc : regst_descs) {
        auto* edge = new SharableMemBlockEdge();
        AddAllocatedEdge(edge);
        Connect(parent, edge, regst_desc2node.at(regst_desc));
      }
    });
  }
}

void SharableMemBlockGraph::ForEachSourceNodeGroup(
    const std::function<int64_t(const SharableMemBlockNode*)>& GroupBy,
    const std::function<void(const std::vector<const SharableMemBlockNode*>&)>& Handler) const {
  HashMap<int64_t, std::vector<const SharableMemBlockNode*>> group_key2source_nodes;
  for (const SharableMemBlockNode* source : source_nodes()) {
    group_key2source_nodes[GroupBy(source)].push_back(source);
  }
  for (const auto& pair : group_key2source_nodes) {
    if (pair.second.size() > 1) { Handler(pair.second); }
  }
}

}  // namespace oneflow
