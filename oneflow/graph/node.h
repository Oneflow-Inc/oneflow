#ifndef ONEFLOW_GRAPH_NODE_H_
#define ONEFLOW_GRAPH_NODE_H_

#include <stdint.h>
#include <unordered_set>
#include <vector>
#include <memory>
#include "common/util.h"

namespace oneflow {

class Node;

class Edge {
 public:
  DISALLOW_COPY_AND_MOVE(Edge);
  Edge() = default;
  virtual ~Edge() = default;

  virtual void Init() {
    src_node_ = nullptr;
    dst_node_ = nullptr;
  }

  Node* src_node() const { return src_node_; }
  Node* dst_node() const { return dst_node_; }

 private:
  // only follows can write the src_node_ and dst_node_
  class TransfmNode;
  class TransfmEdge;
  friend void Connect(Node* src_node, Edge* edge, Node* dst_node);
  friend void DisConnect(Edge* edge);
  friend void TransfmEdge::ConnectDanglingInEdges(TransfmEdge*);
  friend void TransfmEdge::ConnectDanglingOutEdges(TransfmEdge*);
  
  Node* src_node_;
  Node* dst_node_;

};

class Node {
 public:
  DISALLOW_COPY_AND_MOVE(Node);
  Node() = default;
  virtual ~Node() = default;
  
  virtual void Init();

  int32_t node_id() const { return node_id_; }

  const std::unordered_set<Edge*>& in_edges() const {
    return in_edges_;
  }
  const std::unordered_set<Edge*>& out_edges() const {
    return out_edges_;
  }

  void DisconnectAllEdges();

  bool HasSuccessor(const Node* succ_node) const;

  bool HasPredecessor(const Node* pred_node) const;

 private:
  friend void Connect(Node* src_node, Edge* edge, Node* dst_node);
  friend void DisConnect(Edge* edge);
  int32_t node_id_;

  std::unordered_set<Edge*> in_edges_;
  std::unordered_set<Edge*> out_edges_;

};

} // namespace oneflow

#endif  // ONEFLOW_GRAPH_NODE_H_
