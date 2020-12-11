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
#include <sstream>
#include "oneflow/core/graph/stage_chain_graph.h"
#include "oneflow/core/graph/compute_graph.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/ptr_util.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

namespace {

bool IsSspVariableProxy(const ComputeNode& compute_node) {
  const auto& op_conf = compute_node.op().op_conf();
  return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == "ssp_variable_proxy";
}

Maybe<void> GetBackboneNodes(const HashSet<const ComputeNode*>& compute_nodes,
                             HashSet<const ComputeNode*>* backbone_nodes) {
  std::list<const ComputeNode*> starts;
  {
    const auto& ForEachIn = [&](const ComputeNode* node,
                                const std::function<void(const ComputeNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](const ComputeNode* in_node) {
        if (compute_nodes.count(in_node) > 0) { Handler(in_node); }
      });
    };
    const auto& GetInputSize = [&](const ComputeNode* node) {
      size_t input_size = 0;
      ForEachIn(node, [&](const ComputeNode*) { ++input_size; });
      return input_size;
    };
    for (const ComputeNode* node : compute_nodes) {
      if (GetInputSize(node) > 1) { starts.push_back(node); }
    }
  }
  const auto& ForEachOut = [&](const ComputeNode* node,
                               const std::function<void(const ComputeNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](const ComputeNode* out_node) {
      if (compute_nodes.count(out_node) > 0) { Handler(out_node); }
    });
  };
  ComputeGraph().BfsForEachNode(starts, ForEachOut,
                                [&](const ComputeNode* node) { backbone_nodes->insert(node); });
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> StageChainNode::ForEachBackboneSourceComputeNode(
    const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const {
  HashSet<const ComputeNode*> backbone_nodes;
  JUST(GetBackboneNodes(compute_nodes(), &backbone_nodes));
  const auto& IsSource = [&](const ComputeNode* node) {
    size_t num_inputs = 0;
    for (const auto* edge : node->in_edges()) {
      num_inputs += backbone_nodes.count(edge->src_node());
    }
    return num_inputs == 0;
  };
  for (const auto* node : backbone_nodes) {
    if (!IsSource(node)) { continue; }
    // TODO(lixinqi): Remove this urgly code after the ssp_variable_proxy actor bug fixed
    CHECK_OR_RETURN(!IsSspVariableProxy(*node));
    JUST(DoEach(*node));
  }
  return Maybe<void>::Ok();
}

Maybe<void> StageChainNode::ForEachSourceComputeNode(
    const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const {
  const auto& IsSource = [&](const ComputeNode* node) {
    size_t num_inputs = 0;
    for (const auto* edge : node->in_edges()) {
      num_inputs += compute_nodes().count(edge->src_node());
    }
    return num_inputs == 0;
  };
  for (const auto* node : compute_nodes()) {
    if (!IsSource(node)) { continue; }
    // TODO(lixinqi): Remove this urgly code after the ssp_variable_proxy actor bug fixed
    CHECK_OR_RETURN(!IsSspVariableProxy(*node));
    JUST(DoEach(*node));
  }
  return Maybe<void>::Ok();
}

Maybe<void> StageChainNode::ForEachSinkComputeNode(
    const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const {
  const auto& IsSink = [&](const ComputeNode* node) {
    size_t num_outputs = 0;
    for (const auto* edge : node->out_edges()) {
      num_outputs += compute_nodes().count(edge->dst_node());
    }
    return num_outputs == 0;
  };
  for (const auto* node : compute_nodes()) {
    if (!IsSink(node)) { continue; }
    // TODO(lixinqi): Remove this urgly code after the ssp_variable_proxy actor bug fixed
    CHECK_OR_RETURN(!IsSspVariableProxy(*node));
    JUST(DoEach(*node));
  }
  return Maybe<void>::Ok();
}

std::string StageChainNode::VisualStr() const {
  const auto& storage = *Global<vm::SymbolStorage<ParallelDesc>>::Get();
  const auto& parallel_desc = CHECK_JUST(storage.MaybeGet(parallel_desc_symbol_id()));
  const auto& parallel_conf = parallel_desc.parallel_conf();
  std::stringstream ss;
  ss << "calculation_pass_name: " << calculation_pass_name()
     << "\\nstage_id: " << stage_placement_id() << "\\ndevice_tag: " << parallel_conf.device_tag()
     << "\\ndevice_name: " << Join(parallel_conf.device_name(), ", ")
     << "\\nnum_compute_nodes: " << std::to_string(compute_nodes().size())
     << "\\nstage_buffer_size: " << stage_buffer_size();
  size_t cnt = 0;
  const auto& RenderOpName = [&](const ComputeNode& node) -> Maybe<void> {
    if (cnt < 10) {
      ss << "\\n" << node.op().op_name();
    } else if (cnt == 10) {
      ss << "\\n...";
    } else {
      // display nothing
    }
    ++cnt;
    return Maybe<void>::Ok();
  };
  ss << "\\n----[ first 10 backbone source op names ]----";
  cnt = 0;
  CHECK_JUST(ForEachBackboneSourceComputeNode(RenderOpName));
  ss << "\\n----[ first 10 sink op names ]----";
  cnt = 0;
  CHECK_JUST(ForEachSinkComputeNode(RenderOpName));
  return ss.str();
}

void StageChainEdge::add_lbi(const LogicalBlobId& lbi) {
  lbis_.insert(lbi);
  op_names_.insert(lbi.op_name());
}

Maybe<size_t> StageChainEdge::NumStagePlacementInPath() const {
  CHECK_NOTNULL_OR_RETURN(path_stage_placement_ids_.get());
  return path_stage_placement_ids_->size();
}

Maybe<size_t> StageChainEdge::NumParallelDescInPath() const {
  CHECK_NOTNULL_OR_RETURN(path_parallel_desc_symbol_ids_.get());
  return path_parallel_desc_symbol_ids_->size();
}

void StageChainEdge::add_path_stage_placement_id(int64_t stage_placement_id) {
  if (!path_stage_placement_ids_) { path_stage_placement_ids_.reset(new HashSet<int64_t>()); }
  path_stage_placement_ids_->insert(stage_placement_id);
}

void StageChainEdge::add_path_parallel_desc_symbol_id(int64_t parallel_desc_symbol_id) {
  if (!path_parallel_desc_symbol_ids_) {
    path_parallel_desc_symbol_ids_.reset(new HashSet<int64_t>());
  }
  path_parallel_desc_symbol_ids_->insert(parallel_desc_symbol_id);
}

std::string StageChainEdge::VisualStr() const {
  std::stringstream ss;
  std::string ret;
  ret += "num_lbis: ";
  ret += std::to_string(lbis().size());
  return ret;
}

Maybe<void> StageChainGraph::Init(const ComputeGraph& compute_graph) {
  JUST(InitNodes(compute_graph));
  JUST(InitEdges(compute_graph));
  get_dot_end_ = []() -> Maybe<std::string> { return std::make_shared<std::string>(); };
  return Maybe<void>::Ok();
}

Maybe<const StageChainNode&> StageChainGraph::StageChainNode4OpName(
    const std::string& op_name) const {
  return *JUST(MapAt(op_name2chain_node_, op_name));
}

Maybe<StageChainNode*> StageChainGraph::MutStageChainNode4OpName(const std::string& op_name) {
  return JUST(MapAt(op_name2chain_node_, op_name));
}

Maybe<void> StageChainGraph::InitNodes(const ComputeGraph& compute_graph) {
  struct StageChainNodeKey {
    int64_t stage_placement_id;
    int64_t parallel_desc_symbol_id;
    std::string calculation_pass_name;
    std::set<const ComputeNode*> ancestors;

    bool operator<(const StageChainNodeKey& that) const {
      if (this->stage_placement_id != that.stage_placement_id) {
        return this->stage_placement_id < that.stage_placement_id;
      }
      if (this->parallel_desc_symbol_id != that.parallel_desc_symbol_id) {
        return this->parallel_desc_symbol_id < that.parallel_desc_symbol_id;
      }
      if (this->calculation_pass_name != that.calculation_pass_name) {
        return this->calculation_pass_name < that.calculation_pass_name;
      }
      return this->ancestors < that.ancestors;
    }
  };

  struct StageChainNodeInfo {
    int64_t max_buffer_size;
    std::shared_ptr<HashSet<const ComputeNode*>> compute_nodes;
  };

  std::function<Maybe<const std::set<const ComputeNode*>&>(const ComputeNode&)>
      OtherStageAncestors4ComputeNode;
  JUST(MakeGetterOtherStageAncestors4ComputeNode(compute_graph, &OtherStageAncestors4ComputeNode));
  std::map<StageChainNodeKey, StageChainNodeInfo> key2chain_node_info;
  JUST(compute_graph.ForEachComputeNode([&](const ComputeNode& compute_node) -> Maybe<void> {
    StageChainNodeKey key = {
        .stage_placement_id = compute_node.scope().Int64("stage_placement_id"),
        .parallel_desc_symbol_id = JUST(compute_node.GetParallelDescSymbolId()),
        .calculation_pass_name = compute_node.scope().scope_proto().calculation_pass_name(),
        .ancestors = JUST(OtherStageAncestors4ComputeNode(compute_node)),
    };
    auto* node_info = &key2chain_node_info[key];
    // Updates max buffer size
    int64_t stage_buffer_size = compute_node.scope().Int64("stage_weight_buffer_size");
    node_info->max_buffer_size = std::max(node_info->max_buffer_size, stage_buffer_size);
    // Collects compute nodes
    auto* group = &node_info->compute_nodes;
    if (!*group) { group->reset(new HashSet<const ComputeNode*>()); }
    (*group)->insert(&compute_node);
    return Maybe<void>::Ok();
  }));
  for (const auto& pair : key2chain_node_info) {
    const auto& info = pair.second;
    auto* stage_chain_node = JUST(StageChainNode::UnsafeNew(
        pair.first.stage_placement_id, pair.first.parallel_desc_symbol_id, info.max_buffer_size,
        pair.first.calculation_pass_name, info.compute_nodes));
    AddAllocatedNode(stage_chain_node);
    for (const auto* compute_node : *info.compute_nodes) {
      const auto& op_name = compute_node->op().op_name();
      CHECK_OR_RETURN(op_name2chain_node_.emplace(op_name, stage_chain_node).second);
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
  JUST(compute_graph.MaybeTopoForEachNode([&](const ComputeNode* compute_node) -> Maybe<void> {
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

Maybe<void> StageChainGraph::InitEdges(const ComputeGraph& compute_graph) {
  std::function<StageChainEdge*(StageChainNode * src, StageChainNode * dst)> FindOrCreateEdge;
  MakeGetterFindOrCreateEdge(&FindOrCreateEdge);
  JUST(compute_graph.ForEachComputeNode([&](const ComputeNode& compute_node) -> Maybe<void> {
    const auto& op = compute_node.op();
    auto* cur_stage_chain_node = JUST(MutStageChainNode4OpName(op.op_name()));
    for (const auto& ibn : op.input_bns()) {
      const auto& lbi = op.BnInOp2Lbi(ibn);
      auto* input_stage_chain_node = JUST(MutStageChainNode4OpName(lbi.op_name()));
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

Maybe<void> StageChainGraph::InitEdgeStatistics() {
  auto IsReachable = MakePredicatorIsReachable();
  const auto& UpdateEdgeStatistics = [&](StageChainEdge* edge, StageChainNode* node) {
    edge->add_path_stage_placement_id(node->stage_placement_id());
    edge->add_path_parallel_desc_symbol_id(node->parallel_desc_symbol_id());
  };
  ForEachEdge([&](StageChainEdge* edge) {
    auto* src = edge->src_node();
    auto* dst = edge->dst_node();
    const auto& ForEachNext = [&](StageChainNode* node,
                                  const std::function<void(StageChainNode*)>& DoEach) {
      node->ForEachNodeOnInOutEdge([&](StageChainNode* next) {
        if (IsReachable(src, next) && IsReachable(next, dst)) { DoEach(next); }
      });
    };
    BfsForEachNode({src}, ForEachNext,
                   [&](StageChainNode* node) { UpdateEdgeStatistics(edge, node); });
    UpdateEdgeStatistics(edge, dst);
  });
  return Maybe<void>::Ok();
}

Maybe<void> StageChainGraph::WithDotEndGetter(
    const std::function<Maybe<std::string>()>& get_dot_end,
    const std::function<Maybe<void>()>& Do) {
  get_dot_end_ = get_dot_end;
  JUST(Do());
  get_dot_end_ = []() -> Maybe<std::string> { return std::make_shared<std::string>(); };
  return Maybe<void>::Ok();
}

Maybe<std::string> StageChainGraph::VirtualDotEnd() const { return get_dot_end_(); }

}  // namespace oneflow
