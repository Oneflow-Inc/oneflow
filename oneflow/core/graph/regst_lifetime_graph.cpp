#include "oneflow/core/graph/regst_lifetime_graph.h"

namespace oneflow {

RegstLifetimeGraph::RegstLifetimeGraph(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds) {
  std::list<RegstLifetimeNode*> nodes;
  InitNodes(regst_descs, ComputeLifetimeActorIds, &nodes);
  InitEdges(nodes);
}

void RegstLifetimeGraph::InitNodes(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds,
    std::list<RegstLifetimeNode*>* nodes) {
  HashMap<int64_t, RegstLifetimeNode*> regst_desc_id2regst_lifetime_node;
  HashMap<int64_t, std::vector<int64_t>> regst_desc_id2referencing_regst_desc_ids;
  for (const RegstDescProto* regst_desc : regst_descs) {
    auto lifetime_actor_ids = std::make_unique<HashSet<int64_t>>();
    ComputeLifetimeActorIds(regst_desc, lifetime_actor_ids.get());
    auto* node = new RegstLifetimeNode(regst_desc, std::move(lifetime_actor_ids));
    CHECK(regst_desc_id2regst_lifetime_node.emplace(regst_desc->regst_desc_id(), node).second);
    if (regst_desc->reference_regst_desc_id() != -1) {
      regst_desc_id2referencing_regst_desc_ids[regst_desc->reference_regst_desc_id()].emplace_back(
          regst_desc->regst_desc_id());
    }
    AddAllocatedNode(node);
    nodes->push_back(node);
  }
  for (auto& pair : regst_desc_id2referencing_regst_desc_ids) {
    std::vector<int64_t> regst_desc_id_group = pair.second;
    HashSet<const RegstLifetimeNode*> regst_lifetime_node_group;
    regst_desc_id_group.emplace_back(pair.first);
    for (auto regst_desc_id : regst_desc_id_group) {
      auto regst_lifetime_node_it = regst_desc_id2regst_lifetime_node.find(regst_desc_id);
      CHECK(regst_lifetime_node_it != regst_desc_id2regst_lifetime_node.end());
      CHECK(regst_lifetime_node_group.emplace(regst_lifetime_node_it->second).second);
    }
    for (auto& regst_lifetime_node : regst_lifetime_node_group) {
      CHECK(regst_lifetime_node2reference_nodes_
                .emplace(regst_lifetime_node, regst_lifetime_node_group)
                .second);
    }
  }
}

void RegstLifetimeGraph::InitEdges(const std::list<RegstLifetimeNode*>& nodes) {
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
    const std::function<void(const std::list<const RegstDescProto*>&)>& Handler) const {
  HashMap<const RegstLifetimeNode*, std::set<int32_t>> node2excluded_color_ids;
  HashMap<const RegstLifetimeNode*, int32_t> node2color_id;
  auto ForEachIntersected = &RegstLifetimeNode::ForEachNodeOnInOutEdge;
  ForEachNode([&](const RegstLifetimeNode* start) {
    if (node2color_id.find(start) != node2color_id.end()) { return; }
    BfsForEachNode({start}, ForEachIntersected, [&](const RegstLifetimeNode* node) {
      if (node2color_id.find(node) != node2color_id.end()) { return; }
      int32_t color_id = 0;
      const auto& excluded_color_ids = node2excluded_color_ids[node];
      for (; excluded_color_ids.find(color_id) != excluded_color_ids.end(); ++color_id) {}
      HashSet<const RegstLifetimeNode*> reference_nodes;
      auto reference_it = regst_lifetime_node2reference_nodes_.find(node);
      if (reference_it != regst_lifetime_node2reference_nodes_.end()) {
        reference_nodes = reference_it->second;
      } else {
        CHECK(reference_nodes.emplace(node).second);
      }
      for (auto& reference_node : reference_nodes) {
        node2color_id[reference_node] = color_id;
        (reference_node->*ForEachIntersected)([&](const RegstLifetimeNode* intersected) {
          if (node2color_id.find(intersected) != node2color_id.end()) { return; }
          node2excluded_color_ids[intersected].insert(color_id);
        });
      }

    });
  });
  HashMap<int32_t, std::list<const RegstDescProto*>> color_id2regst_descs;
  for (const auto& pair : node2color_id) {
    color_id2regst_descs[pair.second].push_back(&pair.first->regst_desc());
  }
  for (const auto& pair : color_id2regst_descs) { Handler(pair.second); }
}

}  // namespace oneflow
