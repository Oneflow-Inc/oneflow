#ifndef ONEFLOW_GRAPH_NODE_H_
#define ONEFLOW_GRAPH_NODE_H_

#include <stdint.h>
#include <unordered_set>
#include <vector>
#include <memory>
#include "common/util.h"

namespace oneflow {

template<typename DerivedEdge>
void Connect(typename DerivedEdge::NodeType* src_node,
             DerivedEdge* edge,
             typename DerivedEdge::NodeType* dst_node) {
  src_node->out_edges_.insert(edge);
  dst_node->in_edges_.insert(edge);
  edge->src_node_ = src_node;
  edge->dst_node_ = dst_node;
}

template<typename DerivedEdge>
void DisConnect(DerivedEdge* edge) {
  edge->src_node_->out_edges_.erase(edge);
  edge->dst_node_->in_edges_.erase(edge);
  edge->src_node_ = nullptr;
  edge->dst_node_ = nullptr;
}

int32_t NewNodeId();

template<typename DerivedEdge>
class Edge {
 public:
  DISALLOW_COPY_AND_MOVE(Edge);
  Edge() = default;
  virtual ~Edge() = default;
  
  using NodeType = typename DerivedEdge::NodeType;
  using EdgeType = typename NodeType::EdgeType;
  static_assert(std::is_same<DerivedEdge, EdgeType>::value, "");

  virtual void Init() {
    src_node_ = nullptr;
    dst_node_ = nullptr;
  }

  NodeType* src_node() const { return src_node_; }
  NodeType* dst_node() const { return dst_node_; }

 private:
  friend void Connect<EdgeType>(NodeType* src_node,
                                EdgeType* edge,
                                NodeType* dst_node);
  friend void DisConnect<EdgeType>(EdgeType* edge);
  
  NodeType* src_node_;
  NodeType* dst_node_;

};


template<typename DerivedNode>
class Node {
 public:
  DISALLOW_COPY_AND_MOVE(Node);
  Node() = default;
  virtual ~Node() = default;
  
  using EdgeType = typename DerivedNode::EdgeType;
  using NodeType = typename EdgeType::NodeType;
  static_assert(std::is_same<DerivedNode, NodeType>::value, "");

  virtual void Init() {
    node_id_ = NewNodeId();
  }

  int32_t node_id() const { return node_id_; }

  const std::unordered_set<EdgeType*>& in_edges() const {
    return in_edges_;
  }
  const std::unordered_set<EdgeType*>& out_edges() const {
    return out_edges_;
  }

  void DisconnectAllEdges() {
    for (EdgeType* edge : in_edges_) {
      DisConnect(edge);
    }
    for (EdgeType* edge : out_edges_) {
      DisConnect(edge);
    }
    CHECK(in_edges_.empty());
    CHECK(out_edges_.empty());
  }

  bool HasSuccessor(const NodeType* succ_node) const {
    for (EdgeType* edge : out_edges_) {
      if (edge->dst_node() == succ_node) {
        return true;
      }
    }
    return false;
  }

  bool HasPredecessor(const Node* pred_node) const {
    for (EdgeType* edge : in_edges_) {
      if (edge->src_node() == pred_node) {
        return true;
      }
    }
    return false;
  }

 private:
  friend void Connect<EdgeType>(NodeType* src_node,
                                EdgeType* edge,
                                NodeType* dst_node);
  friend void DisConnect<EdgeType>(EdgeType* edge);

  int32_t node_id_;

  std::unordered_set<EdgeType*> in_edges_;
  std::unordered_set<EdgeType*> out_edges_;

};

} // namespace oneflow

#endif  // ONEFLOW_GRAPH_NODE_H_
