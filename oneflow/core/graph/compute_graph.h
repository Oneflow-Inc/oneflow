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
#ifndef ONEFLOW_CORE_GRAPH_COMPUTE_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_COMPUTE_GRAPH_H_

#include <utility>
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/ptr_util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

class ComputeEdge;
class ComputeGraph;
class OperatorConf;
class Operator;
class ParallelDesc;
class Scope;
class Job;

class ComputeNode final : public Node<ComputeNode, ComputeEdge> {
 public:
  ComputeNode(const ComputeNode&) = delete;
  ComputeNode(ComputeNode&&) = delete;
  ~ComputeNode() = default;

  template<typename... Args>
  static Maybe<ComputeNode*> UnsafeNew(Args&&... args) {
    auto* node = new ComputeNode();
    JUST(node->Init(std::forward<Args>(args)...));
    return node;
  }

  const Operator& op() const { return *op_; }
  const ParallelDesc& parallel_desc() const { return *parallel_desc_; }
  const Scope& scope() const { return *scope_; }

  Maybe<int64_t> GetParallelDescSymbolId() const;
  Maybe<int64_t> topo_order_value() const { return PtrGet(topo_order_value_); }

  std::string VisualStr() const override;

 private:
  friend class ComputeGraph;
  ComputeNode() = default;
  Maybe<void> Init(const OperatorConf&);

  std::shared_ptr<Scope> scope_;
  std::shared_ptr<ParallelDesc> parallel_desc_;
  std::shared_ptr<Operator> op_;
  std::unique_ptr<int64_t> topo_order_value_;
};

class ComputeEdge final : public Edge<ComputeNode, ComputeEdge> {
 public:
  ComputeEdge(const ComputeEdge&) = delete;
  ComputeEdge(ComputeEdge&&) = delete;
  ComputeEdge() = default;
  ~ComputeEdge() = default;
};

class ComputeGraph final : public Graph<const ComputeNode, const ComputeEdge> {
 public:
  ComputeGraph(const ComputeGraph&) = delete;
  ComputeGraph(ComputeGraph&&) = delete;
  ComputeGraph() = default;
  ~ComputeGraph() = default;

  template<typename... Args>
  static Maybe<ComputeGraph> New(Args&&... args) {
    auto node = std::make_shared<ComputeGraph>();
    JUST(node->Init(std::forward<Args>(args)...));
    return node;
  }

  Maybe<void> ForEachComputeNode(
      const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const;
  Maybe<void> TopoForEachComputeNode(
      const std::function<Maybe<void>(const ComputeNode&)>& DoEach) const;

  Maybe<const ComputeNode&> Node4OpName(const std::string& op_name) const {
    return *JUST(MapAt(op_name2node_, op_name));
  }

  void ForEachDataAndCtrlInNode(const ComputeNode* node,
                                const std::function<void(const ComputeNode*)>& Handler) const;
  void ForEachDataAndCtrlOutNode(const ComputeNode* node,
                                 const std::function<void(const ComputeNode*)>& Handler) const;
  Maybe<void> InitTopoOrderValue();

  Maybe<void> WithDotEndGetter(const std::function<Maybe<std::string>()>& get_dot_end,
                               const std::function<Maybe<void>()>& Do);

  Maybe<std::string> VirtualDotEnd() const override;

 private:
  Maybe<void> Init(const Job&);
  Maybe<void> InitNodes(const Job&);
  Maybe<void> InitEdges(const Job&);

  Maybe<ComputeNode*> MutNode4OpName(const std::string& op_name);

  std::list<std::string> op_names_;
  HashMap<std::string, ComputeNode*> op_name2node_;
  HashMap<std::string, std::list<const ComputeNode*>> producer_op_name2ctrl_consumer_nodes_;
  std::function<Maybe<std::string>()> get_dot_end_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COMPUTE_GRAPH_H_
