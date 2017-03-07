#include "node.h"
#include "glog/logging.h"

namespace oneflow {

void Node::Init() {
  static int32_t node_id_cnt = 0;
  node_id_ = node_id_cnt++;
}
  
void Node::DisconnectAllEdges() {
  for (Edge* edge : in_edges_) {
    DisConnect(edge);
  }
  for (Edge* edge : out_edges_) {
    DisConnect(edge);
  }
  CHECK(in_edges_.empty());
  CHECK(out_edges_.empty());
}

bool Node::HasSuccessor(const Node* succ_node) const {
  for (Edge* edge : out_edges_) {
    if (edge->dst_node() == succ_node) {
      return true;
    }
  }
  return false;
}

bool Node::HasPredecessor(const Node* pred_node) const {
  for (Edge* edge : in_edges_) {
    if (edge->src_node() == pred_node) {
      return true;
    }
  }
  return false;
}

void DisConnect(Edge* edge) {
  edge->src_node_->out_edges_.erase(edge);
  edge->dst_node_->in_edges_.erase(edge);
  edge->src_node_ = nullptr;
  edge->dst_node_ = nullptr;
}

void Connect(Node* src_node, Edge* edge, Node* dst_node) {
  src_node->out_edges_.insert(edge);
  dst_node->in_edges_.insert(edge);
  edge->src_node_ = src_node;
  edge->dst_node_ = dst_node;
}

} // namespace oneflow
