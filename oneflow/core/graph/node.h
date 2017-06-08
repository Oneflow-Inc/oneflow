#ifndef ONEFLOW_CORE_GRAPH_NODE_H_
#define ONEFLOW_CORE_GRAPH_NODE_H_

#include <stdint.h>
#include <unordered_set>
#include <vector>
#include <memory>
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
void Connect(NodeType* src_node,
             EdgeType* edge,
             NodeType* dst_node) {
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

uint64_t NewNodeId();
uint64_t NewEdgeId();

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

  uint64_t edge_id() const { return edge_id_; }
  std::string edge_id_str() const { return std::to_string(edge_id_); }

  NodeType* src_node() const { return src_node_; }
  NodeType* dst_node() const { return dst_node_; }

  virtual std::string VisualStr() const { return ""; }

 private:
  friend void Connect<NodeType, EdgeType>(NodeType* src_node,
                                          EdgeType* edge,
                                          NodeType* dst_node);
  friend void DisConnect<EdgeType>(EdgeType* edge);
  
  uint64_t edge_id_;
  
  NodeType* src_node_;
  NodeType* dst_node_;

};

template<typename NodeType, typename EdgeType>
class Node {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Node);
  Node() {
    node_id_ = NewNodeId();
  }
  virtual ~Node() = default;

  uint64_t node_id() const { return node_id_; }
  std::string node_id_str() const { return std::to_string(node_id_); }
  EdgeType* SoleInEdge() const {
    CHECK_EQ(in_edges_.size(), 1);
    return *(in_edges_.begin());
  }
  EdgeType* SoleOutEdge() const {
    CHECK_EQ(out_edges_.size(), 1);
    return *(out_edges_.begin());
  }

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
  }
  
  virtual std::string VisualStr() const { return ""; }

 private:
  friend void Connect<NodeType, EdgeType>(NodeType* src_node,
                                          EdgeType* edge,
                                          NodeType* dst_node);
  friend void DisConnect<EdgeType>(EdgeType* edge);

  uint64_t node_id_;
  std::unordered_set<EdgeType*> in_edges_;
  std::unordered_set<EdgeType*> out_edges_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_NODE_H_
