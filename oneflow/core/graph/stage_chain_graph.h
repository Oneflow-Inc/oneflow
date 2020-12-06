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
#ifndef ONEFLOW_CORE_GRAPH_STAGE_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_STAGE_CHAIN_GRAPH_H_

#include <utility>
#include <memory>
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

class StageChainEdge;
class StageChainGraph;
class ComputeNode;
class ComputeGraph;

class StageChainNode : public Node<StageChainNode, StageChainEdge> {
 public:
  StageChainNode(const StageChainNode&) = delete;
  StageChainNode(StageChainNode&&) = delete;
  ~StageChainNode() = default;

  template<typename... Args>
  static Maybe<StageChainNode*> UnsafeNew(Args&&... args) {
    auto* stage_op_node = new StageChainNode();
    JUST(stage_op_node->Init(std::forward<Args>(args)...));
    return stage_op_node;
  }

  int64_t stage_placement_id() const { return stage_placement_id_; }
  int64_t parallel_desc_symbol_id() const { return parallel_desc_symbol_id_; }
  int64_t stage_buffer_size() const { return stage_buffer_size_; }
  const std::string& calculation_pass_name() const { return calculation_pass_name_; }

  const HashSet<const ComputeNode*>& compute_nodes() const { return *compute_nodes_; }

  Maybe<void> ForEachSourceComputeNode(
      const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const;

  std::string VisualStr() const override;

 private:
  StageChainNode() = default;
  Maybe<void> Init(int64_t stage_placement_id, int64_t parallel_desc_symbol_id,
                   int64_t stage_buffer_size, const std::string& calculation_pass_name,
                   const std::shared_ptr<HashSet<const ComputeNode*>>& compute_nodes) {
    stage_placement_id_ = stage_placement_id;
    parallel_desc_symbol_id_ = parallel_desc_symbol_id;
    stage_buffer_size_ = stage_buffer_size;
    calculation_pass_name_ = calculation_pass_name;
    compute_nodes_ = compute_nodes;
    return Maybe<void>::Ok();
  }

  int64_t stage_placement_id_;
  int64_t parallel_desc_symbol_id_;
  int64_t stage_buffer_size_;
  std::string calculation_pass_name_;
  std::shared_ptr<HashSet<const ComputeNode*>> compute_nodes_;
};

class StageChainEdge : public Edge<StageChainNode, StageChainEdge> {
 public:
  StageChainEdge(const StageChainEdge&) = delete;
  StageChainEdge(StageChainEdge&&) = delete;
  StageChainEdge() = default;
  ~StageChainEdge() = default;

  const HashSet<LogicalBlobId>& lbis() const { return lbis_; }

  void add_lbi(const LogicalBlobId& lbi) { lbis_.insert(lbi); }

  std::string VisualStr() const override;

 private:
  HashSet<LogicalBlobId> lbis_;
};

class StageChainGraph : public Graph<StageChainNode, StageChainEdge> {
 public:
  StageChainGraph(const StageChainGraph&) = delete;
  StageChainGraph(StageChainGraph&&) = delete;
  StageChainGraph() = default;
  ~StageChainGraph() = default;

  template<typename... Args>
  static Maybe<StageChainGraph> New(Args&&... args) {
    auto graph = std::make_shared<StageChainGraph>();
    JUST(graph->Init(std::forward<Args>(args)...));
    return graph;
  }

  Maybe<void> MakeGetterPathStagePlacementIds4Edge(
      std::function<Maybe<const HashSet<int64_t>&>(const StageChainEdge*)>*
          PathStagePlacementIds4Edge) const;

 private:
  Maybe<void> Init(const ComputeGraph& compute_graph);
  Maybe<void> InitNodes(
      const ComputeGraph& compute_graph,
      std::function<Maybe<StageChainNode*>(const std::string&)>* StageChainNode4OpName);
  Maybe<void> InitEdges(
      const ComputeGraph& compute_graph,
      const std::function<Maybe<StageChainNode*>(const std::string&)>& StageChainNode4OpName);

  void MakeGetterFindOrCreateEdge(
      std::function<StageChainEdge*(StageChainNode* src, StageChainNode* dst)>* FindOrCreateEdge);

  Maybe<void> MakeGetterOtherStageAncestors4ComputeNode(
      const ComputeGraph& compute_graph,
      std::function<Maybe<const std::set<const ComputeNode*>&>(const ComputeNode&)>*
          OtherStageAncestors4ComputeNode) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STAGE_CHAIN_GRAPH_H_
