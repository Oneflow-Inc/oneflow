#include "oneflow/core/graph/regst_lifetime_poset_graph.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

RegstLifetimePosetGraph::RegstLifetimePosetGraph(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds) {
  InitNodesAndEdges(regst_descs, ComputeLifetimeActorIds);
  InitRegstLifetimePosetNode2IntersectedNodes();
}

void RegstLifetimePosetGraph::InitNodesAndEdges(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds) {
  // init nodes
  std::list<RegstLifetimePosetNode*> nodes;
  for (const RegstDescProto* regst_desc : regst_descs) {
    auto lifetime_actor_ids = std::make_unique<HashSet<int64_t>>();
    ComputeLifetimeActorIds(regst_desc, lifetime_actor_ids.get());
    auto* node = new RegstLifetimePosetNode(regst_desc, std::move(lifetime_actor_ids));
    AddAllocatedNode(node);
    nodes.push_back(node);
  }
  // init edges
  HashMap<const RegstLifetimePosetNode*, size_t> node2regst_size;
  for (const auto* node : nodes) {
    node2regst_size[node] = RtRegstDesc(node->regst_desc()).packed_blob_desc()->TotalByteSize();
  }
  auto HasPartialOrder = [&](const RegstLifetimePosetNode* src, const RegstLifetimePosetNode* dst) {
    if (src == dst) { return false; }
    if (LifetimeContain(dst, src)) {
      if (!LifetimeContain(src, dst)) { return true; }
      if (node2regst_size.at(src) < node2regst_size.at(dst)) { return true; }
      if (node2regst_size.at(src) == node2regst_size.at(dst)) { return src < dst; }
    }
    return false;
  };
  for (RegstLifetimePosetNode* src : nodes) {
    for (RegstLifetimePosetNode* dst : nodes) {
      if (HasPartialOrder(src, dst)) { Connect(src, NewEdge(), dst); }
    }
  }
}

bool RegstLifetimePosetGraph::LifetimeContain(
    const RegstLifetimePosetNode* long_lifetime_node,
    const RegstLifetimePosetNode* short_lifetime_node) const {
  for (int64_t actor_id : short_lifetime_node->lifetime_actor_ids()) {
    if (long_lifetime_node->lifetime_actor_ids().find(actor_id)
        == long_lifetime_node->lifetime_actor_ids().end()) {
      return false;
    }
  }
  return true;
}

void RegstLifetimePosetGraph::InitRegstLifetimePosetNode2IntersectedNodes() {
  HashMap<int64_t, HashSet<const RegstLifetimePosetNode*>> actor_id2nodes;
  ForEachNode([&](const RegstLifetimePosetNode* node) {
    for (int64_t actor_id : node->lifetime_actor_ids()) { actor_id2nodes[actor_id].insert(node); }
  });
  for (const auto& pair : actor_id2nodes) {
    for (const RegstLifetimePosetNode* node : pair.second) {
      regst_lifetime_node2intersected_nodes_[node].insert(pair.second.begin(), pair.second.end());
    }
  }
  for (auto& pair : regst_lifetime_node2intersected_nodes_) { pair.second.erase(pair.first); }
}

void RegstLifetimePosetGraph::ForEachSameColoredRegstDescs(
    const HashSet<const RegstLifetimePosetNode*>& layer_nodes,
    const std::function<void(const std::list<const RegstDescProto*>&)>& Handler) const {
  auto ForEachIntersected = [&](const RegstLifetimePosetNode* node,
                                const std::function<void(const RegstLifetimePosetNode*)>& Handler) {
    for (const auto* intersected_node : regst_lifetime_node2intersected_nodes_.at(node)) {
      if (layer_nodes.find(intersected_node) != layer_nodes.end()) { Handler(intersected_node); }
    }
  };
  HashMap<const RegstLifetimePosetNode*, std::set<int32_t>> node2excluded_color_ids;
  HashMap<const RegstLifetimePosetNode*, int32_t> node2color_id;
  for (const RegstLifetimePosetNode* start : layer_nodes) {
    if (node2color_id.find(start) != node2color_id.end()) { continue; }
    BfsForEachNode({start}, ForEachIntersected, [&](const RegstLifetimePosetNode* node) {
      if (node2color_id.find(node) != node2color_id.end()) { return; }
      int32_t color_id = 0;
      const auto& excluded_color_ids = node2excluded_color_ids.at(node);
      for (; excluded_color_ids.find(color_id) != excluded_color_ids.end(); ++color_id) {}
      node2color_id[node] = color_id;
      ForEachIntersected(node, [&](const RegstLifetimePosetNode* intersected) {
        if (node2color_id.find(intersected) != node2color_id.end()) { return; }
        node2excluded_color_ids[intersected].insert(color_id);
      });
    });
  }
  HashMap<int32_t, std::list<const RegstDescProto*>> color_id2regst_descs;
  for (const auto& pair : node2color_id) {
    color_id2regst_descs[pair.second].push_back(&pair.first->regst_desc());
  }
  for (const auto& pair : color_id2regst_descs) { Handler(pair.second); }
}

void RegstLifetimePosetGraph::ForEachLayerwiseSameColoredRegstDescs(
    const std::function<void(const std::list<const RegstDescProto*>&)>& Handler) const {
  HashSet<const RegstLifetimePosetNode*> remainder_nodes;
  ForEachNode([&](const RegstLifetimePosetNode* node) { remainder_nodes.insert(node); });
  auto IsSourceNode = [&](const RegstLifetimePosetNode* node) -> bool {
    size_t num = 0;
    node->ForEachNodeOnInEdge([&](const RegstLifetimePosetNode* in_node) {
      if (remainder_nodes.find(in_node) != remainder_nodes.end()) { ++num; }
    });
    return num == 0;
  };
  while (!remainder_nodes.empty()) {
    HashSet<const RegstLifetimePosetNode*> cur_layer_nodes;
    for (const RegstLifetimePosetNode* node : remainder_nodes) {
      if (IsSourceNode(node)) { cur_layer_nodes.insert(node); }
    }
    ForEachSameColoredRegstDescs(cur_layer_nodes, Handler);
    for (const auto* node : cur_layer_nodes) { remainder_nodes.erase(node); }
  }
}

}  // namespace oneflow
