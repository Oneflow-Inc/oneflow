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
#include "oneflow/core/graph/regst_lifetime_graph.h"

namespace oneflow {

RegstLifetimeGraph::RegstLifetimeGraph(
    const std::vector<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds) {
  std::vector<RegstLifetimeNode*> nodes;
  InitNodes(regst_descs, ComputeLifetimeActorIds, &nodes);
  InitEdges(nodes);
}

void RegstLifetimeGraph::InitNodes(
    const std::vector<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds,
    std::vector<RegstLifetimeNode*>* nodes) {
  for (const RegstDescProto* regst_desc : regst_descs) {
    auto lifetime_actor_ids = std::make_unique<HashSet<int64_t>>();
    ComputeLifetimeActorIds(regst_desc, lifetime_actor_ids.get());
    auto* node = new RegstLifetimeNode(regst_desc, std::move(lifetime_actor_ids));
    AddAllocatedNode(node);
    nodes->push_back(node);
  }
}

void RegstLifetimeGraph::InitEdges(const std::vector<RegstLifetimeNode*>& nodes) {
  HashMap<int64_t, HashSet<RegstLifetimeNode*>> task_id2intersected_nodes;
  for (RegstLifetimeNode* node : nodes) {
    for (int64_t task_id : node->lifetime_actor_ids()) {
      task_id2intersected_nodes[task_id].insert(node);
    }
  }
  HashMap<RegstLifetimeNode*, HashSet<RegstLifetimeNode*>> src_node2dst_nodes;
  for (const auto& pair : task_id2intersected_nodes) {
    for (RegstLifetimeNode* src_node : pair.second) {
      for (RegstLifetimeNode* dst_node : pair.second) {
        if (src_node->regst_desc_id() < dst_node->regst_desc_id()) {
          src_node2dst_nodes[src_node].emplace(dst_node);
        }
      }
    }
  }
  for (const auto& pair : src_node2dst_nodes) {
    for (RegstLifetimeNode* dst_node : pair.second) { Connect(pair.first, NewEdge(), dst_node); }
  }
}

void RegstLifetimeGraph::ForEachSameColoredRegstDescs(
    const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) const {
  std::vector<const RegstLifetimeNode*> nodes;
  ForEachNode([&](const RegstLifetimeNode* node) { nodes.push_back(node); });
  std::sort(nodes.begin(), nodes.end(),
            [&](const RegstLifetimeNode* lhs, const RegstLifetimeNode* rhs) {
              return lhs->byte_size() > rhs->byte_size();
            });
  HashMap<const RegstLifetimeNode*, std::set<int32_t>> node2excluded_color_ids;
  HashMap<const RegstLifetimeNode*, int32_t> node2color_id;
  for (const RegstLifetimeNode* node : nodes) {
    int32_t color_id = 0;
    const auto& excluded_color_ids = node2excluded_color_ids[node];
    for (; excluded_color_ids.find(color_id) != excluded_color_ids.end(); ++color_id) {}
    node2color_id[node] = color_id;
    node->ForEachNodeOnInOutEdge([&](const RegstLifetimeNode* intersected) {
      if (node2color_id.find(intersected) != node2color_id.end()) { return; }
      node2excluded_color_ids[intersected].insert(color_id);
    });
  }
  HashMap<int32_t, std::vector<const RegstDescProto*>> color_id2regst_descs;
  for (const auto& pair : node2color_id) {
    color_id2regst_descs[pair.second].push_back(&pair.first->regst_desc());
  }
  for (const auto& pair : color_id2regst_descs) { Handler(pair.second); }
}

}  // namespace oneflow
