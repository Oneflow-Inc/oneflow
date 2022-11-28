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
#ifndef ONEFLOW_CORE_GRAPH_NODE_H_
#define ONEFLOW_CORE_GRAPH_NODE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
void Connect(NodeType* src_node, EdgeType* edge, NodeType* dst_node) {
  CHECK(src_node->out_edges_.insert(edge).second);
  CHECK(dst_node->in_edges_.insert(edge).second);
  CHECK(edge->src_node_ == nullptr);
  CHECK(edge->dst_node_ == nullptr);
  edge->src_node_ = src_node;
  edge->dst_node_ = dst_node;
}

template<typename EdgeType>
void DisConnect(EdgeType* edge) {
  CHECK_EQ(edge->src_node_->out_edges_.erase(edge), 1);
  CHECK_EQ(edge->dst_node_->in_edges_.erase(edge), 1);
  edge->src_node_ = nullptr;
  edge->dst_node_ = nullptr;
}

int64_t NewNodeId();
int64_t NewEdgeId();

template<typename NodeType, typename EdgeType>
class Edge {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Edge);
  Edge() {
    edge_id_ = NewEdgeId();
    src_node_ = nullptr;
    dst_node_ = nullptr;
  }
  virtual ~Edge() = default;

  int64_t edge_id() const { return edge_id_; }

  NodeType* src_node() const { return src_node_; }
  NodeType* dst_node() const { return dst_node_; }

  virtual std::string VisualStr() const { return ""; }

 private:
  friend void Connect<NodeType, EdgeType>(NodeType* src_node, EdgeType* edge, NodeType* dst_node);
  friend void DisConnect<EdgeType>(EdgeType* edge);

  int64_t edge_id_;

  NodeType* src_node_;
  NodeType* dst_node_;
};

template<typename NodeType, typename EdgeType>
class Node {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Node);
  Node() { node_id_ = NewNodeId(); }
  virtual ~Node() = default;

  int64_t node_id() const { return node_id_; }
  std::string node_id_str() const { return std::to_string(node_id_); }

  EdgeType* SoleInEdge() const {
    CHECK_EQ(in_edges_.size(), 1);
    return *(in_edges_.begin());
  }
  EdgeType* SoleOutEdge() const {
    CHECK_EQ(out_edges_.size(), 1);
    return *(out_edges_.begin());
  }

  const std::unordered_set<EdgeType*>& in_edges() const { return in_edges_; }
  const std::unordered_set<EdgeType*>& out_edges() const { return out_edges_; }

  void ForEachNodeOnInEdge(std::function<void(NodeType*)> Handler) const {
    for (EdgeType* edge : in_edges_) { Handler(edge->src_node()); }
  }
  void ForEachNodeOnOutEdge(std::function<void(NodeType*)> Handler) const {
    for (EdgeType* edge : out_edges_) { Handler(edge->dst_node()); }
  }
  void ForEachNodeOnInOutEdge(std::function<void(NodeType*)> Handler) const {
    ForEachNodeOnInEdge(Handler);
    ForEachNodeOnOutEdge(Handler);
  }
  Maybe<void> ForEachInNode(std::function<Maybe<void>(NodeType*)> Handler) const {
    for (EdgeType* edge : in_edges_) { JUST(Handler(edge->src_node())); }
    return Maybe<void>::Ok();
  }
  Maybe<void> ForEachOutNode(std::function<Maybe<void>(NodeType*)> Handler) const {
    for (EdgeType* edge : out_edges_) { JUST(Handler(edge->dst_node())); }
    return Maybe<void>::Ok();
  }
  Maybe<void> ForEachInOutNode(std::function<Maybe<void>(NodeType*)> Handler) const {
    JUST(ForEachNodeOnInEdge(Handler));
    JUST(ForEachNodeOnOutEdge(Handler));
    return Maybe<void>::Ok();
  }

  void ForEachNodeOnSortedInEdge(std::function<void(NodeType*)> Handler) const {
    for (EdgeType* edge : sorted_in_edges_) { Handler(edge->src_node()); }
  }
  void ForEachNodeOnSortedOutEdge(std::function<void(NodeType*)> Handler) const {
    for (EdgeType* edge : sorted_out_edges_) { Handler(edge->dst_node()); }
  }
  void ForEachNodeOnSortedInOutEdge(std::function<void(NodeType*)> Handler) const {
    ForEachNodeOnSortedInEdge(Handler);
    ForEachNodeOnSortedOutEdge(Handler);
  }

  void DisconnectAllEdges() {
    for (EdgeType* edge : in_edges_) { DisConnect(edge); }
    for (EdgeType* edge : out_edges_) { DisConnect(edge); }
  }

  virtual std::string VisualStr() const { return ""; }

  void SortInOutEdges(std::function<bool(const EdgeType* lhs, const EdgeType* rhs)> LessThan) {
    sorted_in_edges_.assign(in_edges_.begin(), in_edges_.end());
    sorted_out_edges_.assign(out_edges_.begin(), out_edges_.end());
    std::sort(sorted_in_edges_.begin(), sorted_in_edges_.end(), LessThan);
    std::sort(sorted_out_edges_.begin(), sorted_out_edges_.end(), LessThan);
  }

 private:
  friend void Connect<NodeType, EdgeType>(NodeType* src_node, EdgeType* edge, NodeType* dst_node);
  friend void DisConnect<EdgeType>(EdgeType* edge);

  int64_t node_id_;
  HashSet<EdgeType*> in_edges_;
  HashSet<EdgeType*> out_edges_;
  std::vector<EdgeType*> sorted_in_edges_;
  std::vector<EdgeType*> sorted_out_edges_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NODE_H_
