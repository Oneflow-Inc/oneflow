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
#include "oneflow/core/graph/stage_chain_graph.h"
#include "oneflow/core/graph/compute_graph.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

std::string StageChainNode::VisualStr() const {
  std::string ret;
  for (const auto* compute_node : compute_nodes()) {
    ret += compute_node->op().op_name();
    ret += "\\n";
  }
  return ret;
}

std::string StageChainEdge::VisualStr() const {
  std::string ret;
  for (const auto& lbi : lbis()) {
    ret += GenLogicalBlobName(lbi);
    ret += "\\n";
  }
  return ret;
}

Maybe<void> StageChainGraph::Init(const ComputeGraph& compute_graph) {
  std::function<Maybe<StageChainNode*>(const std::string&)> StageChainNode4OpName;
  JUST(InitNodes(compute_graph, &StageChainNode4OpName));
  JUST(InitEdges(compute_graph, StageChainNode4OpName));
  return Maybe<void>::Ok();
}

Maybe<void> StageChainGraph::InitNodes(
    const ComputeGraph& compute_graph,
    std::function<Maybe<StageChainNode*>(const std::string&)>* StageChainNode4OpName) {
  auto op_name2chain_node = std::make_shared<HashMap<std::string, StageChainNode*>>();
  *StageChainNode4OpName = [op_name2chain_node](const std::string& op_name) {
    return MapAt(*op_name2chain_node, op_name);
  };

  struct StageChaineNodeInfo {
    int64_t max_buffer_size;
    std::shared_ptr<HashSet<int64_t>> parallel_desc_symbol_ids;
    std::shared_ptr<std::list<const ComputeNode*>> compute_nodes;
  };

  std::function<Maybe<const std::set<const ComputeNode*>&>(const ComputeNode&)>
      OtherStageAncestors4ComputeNode;
  JUST(MakeGetterOtherStageAncestors4ComputeNode(compute_graph, &OtherStageAncestors4ComputeNode));
  std::map<std::set<const ComputeNode*>, HashMap<int64_t, StageChaineNodeInfo>>
      other_stage_ancestors2info;
  HashMap<int64_t, StageChaineNodeInfo> stage_placement_id2info;
  JUST(compute_graph.ForEachComputeNode([&](const ComputeNode& compute_node) -> Maybe<void> {
    const auto& ancestors = JUST(OtherStageAncestors4ComputeNode(compute_node));
    auto* stage_placement_id2info = &other_stage_ancestors2info[ancestors];
    int64_t stage_placement_id = compute_node.scope().Int64("stage_placement_id");
    auto* node_info = &(*stage_placement_id2info)[stage_placement_id];
    // Updates max buffer size
    int64_t stage_buffer_size = compute_node.scope().Int64("stage_weight_buffer_size");
    node_info->max_buffer_size = std::max(node_info->max_buffer_size, stage_buffer_size);
    // Collects parallel desc symbol id
    auto* symbol_ids = &node_info->parallel_desc_symbol_ids;
    if (!*symbol_ids) { symbol_ids->reset(new HashSet<int64_t>()); }
    // op's actual parallel_desc_symbol_id may be not device_parallel_desc_symbol_id, but we don't
    // care. What we want here is the device_parallel_desc_symbol_id set by users.
    (*symbol_ids)->insert(compute_node.scope().scope_proto().device_parallel_desc_symbol_id());
    // Collects compute nodes
    auto* group = &node_info->compute_nodes;
    if (!*group) { group->reset(new std::list<const ComputeNode*>()); }
    (*group)->push_back(&compute_node);
    return Maybe<void>::Ok();
  }));
  for (const auto& ancestors7node_info : other_stage_ancestors2info) {
    for (const auto& pair : ancestors7node_info.second) {
      const auto& info = pair.second;
      auto* stage_chain_node = JUST(StageChainNode::UnsafeNew(
          pair.first, info.max_buffer_size, info.parallel_desc_symbol_ids, info.compute_nodes));
      AddAllocatedNode(stage_chain_node);
      for (const auto* compute_node : *info.compute_nodes) {
        const auto& op_name = compute_node->op().op_name();
        CHECK_OR_RETURN(op_name2chain_node->emplace(op_name, stage_chain_node).second);
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> StageChainGraph::MakeGetterOtherStageAncestors4ComputeNode(
    const ComputeGraph& compute_graph,
    std::function<Maybe<const std::set<const ComputeNode*>&>(const ComputeNode&)>*
        OtherStageAncestors4ComputeNode) const {
  using CacheT = HashMap<const ComputeNode*, std::set<const ComputeNode*>>;
  auto node2other_stage_ancestors = std::make_shared<CacheT>();
  JUST(
      compute_graph.TopoForEachNodeWithErrorCaptured([&](ComputeNode* compute_node) -> Maybe<void> {
        auto* cur_other_stage_ancestors = &(*node2other_stage_ancestors)[compute_node];
        int64_t cur_stage_placement_id = compute_node->scope().Int64("stage_placement_id");
        for (auto* edge : compute_node->in_edges()) {
          auto* in_node = edge->src_node();
          if (in_node->scope().Int64("stage_placement_id") == cur_stage_placement_id) {
            const auto& in_ancestors = node2other_stage_ancestors->at(in_node);
            cur_other_stage_ancestors->insert(in_ancestors.begin(), in_ancestors.end());
          } else {
            cur_other_stage_ancestors->insert(in_node);
          }
        }
        return Maybe<void>::Ok();
      }));
  *OtherStageAncestors4ComputeNode =
      [node2other_stage_ancestors](
          const ComputeNode& node) -> Maybe<const std::set<const ComputeNode*>&> {
    return MapAt(*node2other_stage_ancestors, &node);
  };
  return Maybe<void>::Ok();
}

Maybe<void> StageChainGraph::InitEdges(
    const ComputeGraph& compute_graph,
    const std::function<Maybe<StageChainNode*>(const std::string&)>& StageChainNode4OpName) {
  std::function<StageChainEdge*(StageChainNode * src, StageChainNode * dst)> FindOrCreateEdge;
  MakeGetterFindOrCreateEdge(&FindOrCreateEdge);
  JUST(compute_graph.ForEachComputeNode([&](const ComputeNode& compute_node) -> Maybe<void> {
    const auto& op = compute_node.op();
    auto* cur_stage_chain_node = JUST(StageChainNode4OpName(op.op_name()));
    for (const auto& ibn : op.input_bns()) {
      const auto& lbi = op.BnInOp2Lbi(ibn);
      auto* input_stage_chain_node = JUST(StageChainNode4OpName(lbi.op_name()));
      if (input_stage_chain_node == cur_stage_chain_node) { continue; }
      FindOrCreateEdge(input_stage_chain_node, cur_stage_chain_node)->add_lbi(lbi);
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

void StageChainGraph::MakeGetterFindOrCreateEdge(
    std::function<StageChainEdge*(StageChainNode* src, StageChainNode* dst)>* FindOrCreateEdge) {
  using CacheT = HashMap<StageChainNode*, HashMap<StageChainNode*, StageChainEdge*>>;
  auto cache = std::make_shared<CacheT>();
  *FindOrCreateEdge = [cache, this](StageChainNode* src, StageChainNode* dst) {
    StageChainEdge** edge_ptr = &(*cache)[src][dst];
    if (*edge_ptr == nullptr) {
      *edge_ptr = NewEdge();
      Connect(src, *edge_ptr, dst);
    }
    return *edge_ptr;
  };
}

}  // namespace oneflow
